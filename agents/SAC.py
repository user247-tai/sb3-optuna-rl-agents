import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import gymnasium as gym
import stable_baselines3 as sb3
from stable_baselines3.common.callbacks import *
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import stable_baselines3.common.noise as sb3_noise
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import torch.nn as nn
from custom.custom_callbacks import TrialEvalCallback
from utils.env_utils import make_env
from utils.visualize_utils import visualize_optimized_results, advanced_plot_training_result, plot_training_result
import numpy as np

NUM_ENV = 10
ENV_ID = "HalfCheetah-v5"
EVAL_FREQ = max(50_000 // NUM_ENV, 1)
TOTAL_OPTIMIZE_STEPS = 500_000
TOTAL_TRAINING_STEPS = 5_000_000
MODEL_DIR = "trained_model"
MODEL_PATH = os.path.join(MODEL_DIR, f"sac_{ENV_ID}")

def sample_sac_params(trial: optuna.Trial):
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    buffer_size_pow = trial.suggest_int("buffer_size_pow", 4, 6)
    buffer_size = 10 ** buffer_size_pow
    learning_starts_pow = trial.suggest_int("learning_starts_pow", 3, 4)
    learning_starts = 10 ** learning_starts_pow
    batch_size = trial.suggest_categorical("batch_size", [128, 256, 512, 1024])
    tau = trial.suggest_float("tau", 0.001, 0.02, log=True)
    gamma = trial.suggest_float("gamma", 0.95, 0.9999, log=True)
    train_freq_div = trial.suggest_int("train_freq_div", 1, 10)
    train_freq = train_freq_div * NUM_ENV
    gradient_steps = trial.suggest_int("gradient_steps", 1, NUM_ENV)
    n_steps = trial.suggest_int("n_steps", 1, 5)
    ent_coef = trial.suggest_categorical("ent_coef", ["auto", "auto_0.1"])
    use_sde = trial.suggest_categorical("use_sde", [True, False])
    sde_sample_freq = -1
    use_sde_at_warmup = False
    if use_sde:
        sde_sample_freq = trial.suggest_categorical("sde_sample_freq", [8, 16, 32, 64])
        use_sde_at_warmup = trial.suggest_categorical("use_sde_at_warmup", [True, False])
    else:
        trial.set_user_attr("sde_sample_freq", -1)
        trial.set_user_attr("use_sde_at_warmup", False)
    
    activation_fn_name = trial.suggest_categorical("activation_fn", ["tanh", "relu"])
    activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU}[activation_fn_name]

    return {
        "learning_rate": learning_rate,
        "buffer_size": buffer_size,
        "learning_starts": learning_starts,
        "batch_size": batch_size,
        "tau": tau,
        "gamma": gamma,
        "train_freq": train_freq,
        "gradient_steps": gradient_steps,
        "n_steps": n_steps,
        "ent_coef": ent_coef,
        "target_update_interval": 1, # For soft-update
        "target_entropy": "auto",
        "use_sde": use_sde,
        "sde_sample_freq": sde_sample_freq,
        "use_sde_at_warmup": use_sde_at_warmup,
        "policy_kwargs": {
            "activation_fn": activation_fn,
            "net_arch": [256, 256], # Standard MLP architecture for MuJoCo environments.
        },
    }


def objective(trial: optuna.Trial):
    train_env = DummyVecEnv([make_env(ENV_ID, i, "log/sac") for i in range(NUM_ENV)])
    eval_env = DummyVecEnv([lambda: Monitor(gym.make(ENV_ID))])
    kwargs = sample_sac_params(trial)
    
    eval_callback = TrialEvalCallback(
        eval_env,
        trial,
        n_eval_episodes=10,
        eval_freq=EVAL_FREQ,
        deterministic=True,
    )

    model = sb3.SAC("MlpPolicy", train_env, verbose=0, **kwargs)

    nan_encountered = False
    try:
        model.learn(total_timesteps=TOTAL_OPTIMIZE_STEPS, callback=eval_callback, progress_bar=True)
    except AssertionError as e:
        print(e)
        nan_encountered = True
    finally:
        train_env.close()
        eval_env.close()

    if nan_encountered:
        return float("nan")
    
    if eval_callback.is_pruned:
        raise optuna.exceptions.TrialPruned()

    return eval_callback.last_mean_reward


def run_optimization():
    # Create the path to the 'db' subdirectory
    output_dir = "db"
    
    # Ensure the directory exists to avoid FileNotFoundError
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    study_name = f"sac_{ENV_ID}_optimization"
    storage_name = f"sqlite:///{output_dir}/{study_name}.db"
    sampler = TPESampler(n_startup_trials=10, multivariate=True)

    pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=5)
    
    study = optuna.create_study(sampler=sampler, 
                                pruner=pruner, 
                                direction="maximize", 
                                storage=storage_name,
                                study_name=study_name,
                                load_if_exists=True)
    try:
        study.optimize(objective, n_trials=50, timeout=None)
    except KeyboardInterrupt:
        pass

    print(f"Number of finished trials: {len(study.trials)}")

    print("Best trial:")
    trial = study.best_trial
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    print("  User attrs:")
    for key, value in trial.user_attrs.items():
        print(f"    {key}: {value}")

    # Write report
    csv_path = os.path.join(output_dir, f"study_results_sac_{ENV_ID}.csv")
    study.trials_dataframe().to_csv(csv_path, index=False)


def train_agent(log_dir = "log/sac"):
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    train_env = DummyVecEnv([make_env(ENV_ID, i, log_dir) for i in range(NUM_ENV)])
    eval_env = DummyVecEnv([lambda: Monitor(gym.make(ENV_ID))])
    # callback_on_thr = StopTrainingOnRewardThreshold(reward_threshold=500, verbose=1)
    # callback_on_best = StopTrainingOnNoModelImprovement(max_no_improvement_evals=2, min_evals=10, verbose=1)
    eval_callback = EvalCallback(eval_env, n_eval_episodes=50, eval_freq=EVAL_FREQ, verbose=1)
    #eval_callback = EvalCallback(eval_env, n_eval_episodes=50, callback_after_eval=callback_on_best, verbose=1, eval_freq=1000)

    buffer_size = int(10**4)
    learning_starts = int(10**4)
    ent_coef = 0.1 

    best_params = {
        "learning_rate": 0.0008344569983060691,
        "buffer_size": buffer_size,
        "learning_starts": learning_starts,
        "batch_size": 1024,
        "tau": 0.015227923399821753,
        "gamma": 0.9633697103560946,
        "train_freq": 2,
        "gradient_steps": 10,
        "ent_coef": ent_coef,
        "use_sde": True,
        "sde_sample_freq": 16,
        "use_sde_at_warmup": True,
        "policy_kwargs": {
            "activation_fn": nn.Tanh,
            "net_arch": [256, 256], # Standard MLP architecture for MuJoCo environments.
        },
    }

    model = sb3.SAC(
        "MlpPolicy", 
        train_env, 
        verbose=1,
        learning_rate=best_params["learning_rate"],
        buffer_size=best_params["buffer_size"],
        learning_starts=best_params["learning_starts"],
        batch_size=best_params["batch_size"],
        tau=best_params["tau"],
        gamma=best_params["gamma"],
        train_freq=best_params["train_freq"],
        gradient_steps=best_params["gradient_steps"],
        ent_coef=best_params["ent_coef"],
        use_sde=best_params["use_sde"],
        sde_sample_freq=best_params["sde_sample_freq"],
        use_sde_at_warmup=best_params["use_sde_at_warmup"],
        policy_kwargs=best_params["policy_kwargs"],
    )

    model.learn(total_timesteps=TOTAL_TRAINING_STEPS, callback=eval_callback, log_interval=None, progress_bar=True)
    model.save(MODEL_PATH)


def test_agent():
    env = gym.make(ENV_ID, render_mode="human")
    model = sb3.SAC.load(MODEL_PATH, env)
    stop = False
    obs, _ = env.reset()
    eps_return = 0
    while not stop:
        action, _ = model.predict(observation=obs)
        next_obs, reward, terminate, truncate, info = env.step(action)
        eps_return += reward
        obs = next_obs
        stop = terminate or truncate

    print(f"Episode return: {eps_return}")


def main():
    # run_optimization()
    # visualize_optimized_results(f"sac_{ENV_ID}_optimization")
    train_agent()
    # advanced_plot_training_result()
    # test_agent()


if __name__ == "__main__":
    main()
