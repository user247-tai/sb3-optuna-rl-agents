import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import gymnasium as gym
import sb3_contrib
from stable_baselines3.common.callbacks import *
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import torch.nn as nn
from custom.custom_callbacks import TrialEvalCallback
from utils.env_utils import make_env, make_subproc_vec_env
from utils.visualize_utils import visualize_optimized_results, advanced_plot_training_result, plot_training_result
import numpy as np

NUM_ENV = 15
ENV_ID = "HalfCheetah-v4"
EVAL_FREQ = max(200_000 // NUM_ENV, 1)
TOTAL_OPTIMIZE_STEPS = 500_000
TOTAL_TRAINING_STEPS = 5_000_000
MODEL_DIR = "trained_model"
MODEL_PATH = os.path.join(MODEL_DIR, f"ars_{ENV_ID}")

def sample_ars_params(trial: optuna.Trial):    
    n_delta = trial.suggest_int("n_delta", 4, 64, log=True)
    n_top_fraction = trial.suggest_float("n_top_fraction", 0.5, 1.0)
    n_top = max(1, int(n_delta * n_top_fraction))
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 0.1, log=True)
    delta_std = trial.suggest_float("delta_std", 0.01, 0.05, log=True)
    zero_policy = trial.suggest_categorical("zero_policy", [True, False])
    alive_bonus_offset = trial.suggest_float("alive_bonus_offset", 0.0, 5.0)
    
    activation_fn_name = trial.suggest_categorical("activation_fn", ["tanh", "relu"])
    activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU}[activation_fn_name]

    return {
        "n_delta": n_delta,
        "n_top": n_top,
        "learning_rate": learning_rate,
        "delta_std": delta_std,
        "zero_policy": zero_policy,
        "alive_bonus_offset": alive_bonus_offset,
        "policy_kwargs": {
            "activation_fn": activation_fn,
            "net_arch": [256, 256], # Standard MLP architecture for MuJoCo environments.
        },
    }


def objective(trial: optuna.Trial):
    train_env = make_subproc_vec_env(ENV_ID, NUM_ENV, "log/ars")
    eval_env = DummyVecEnv([lambda: Monitor(gym.make(ENV_ID))])
    kwargs = sample_ars_params(trial)
    
    eval_callback = TrialEvalCallback(
        eval_env,
        trial,
        n_eval_episodes=10,
        eval_freq=EVAL_FREQ,
        deterministic=True,
    )

    model = sb3_contrib.ARS("MlpPolicy", train_env, verbose=0, n_eval_episodes=5, **kwargs)

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

    study_name = f"ars_{ENV_ID}_optimization"
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
    csv_path = os.path.join(output_dir, f"study_results_ars_{ENV_ID}.csv")
    study.trials_dataframe().to_csv(csv_path, index=False)


def train_agent(log_dir = "log/ars"):
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

    model = sb3_contrib.ARS(
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
    model = sb3_contrib.ARS.load(MODEL_PATH, env)
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
    run_optimization()
    # visualize_optimized_results(f"ars_{ENV_ID}_optimization")
    # train_agent()
    # advanced_plot_training_result()
    # test_agent()



if __name__ == "__main__":
    main()
