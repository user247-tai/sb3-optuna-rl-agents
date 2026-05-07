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
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import torch.nn as nn
from custom.custom_callbacks import TrialEvalCallback
from utils.env_utils import make_env
from utils.visualize_utils import visualize_optimized_results, advanced_plot_training_result, plot_training_result

NUM_ENV = 10
ENV_ID = "CartPole-v1"
EVAL_FREQ = max(50_000 // NUM_ENV, 1)
TOTAL_OPTIMIZE_STEPS = 100_000
TOTAL_TRAINING_STEPS = 500_000
MODEL_DIR = "trained_model"
MODEL_PATH = os.path.join(MODEL_DIR, f"a2c_{ENV_ID}")


def sample_a2c_params(trial: optuna.Trial):
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 7e-4, log=True)
    n_steps = trial.suggest_categorical("n_steps", [5, 8, 16, 32, 64, 128])
    gamma = trial.suggest_float("gamma", 0.9, 0.9999, log=True)
    gae_lambda = trial.suggest_float("gae_lambda", 0.8, 1.0)
    ent_coef = trial.suggest_float("ent_coef", 0.00001, 0.01, log=True)
    vf_coef = trial.suggest_float("vf_coef", 0.1, 0.9)
    max_grad_norm = trial.suggest_float("max_grad_norm", 0.3, 5.0)
    rms_prop_eps = trial.suggest_float("rms_prop_eps", 1e-06, 1e-04, log=True)
    use_rms_prop = trial.suggest_categorical("use_rms_prop", [True, False])
    # use_sde = trial.suggest_categorical("use_sde", [True, False])
    # sde_sample_freq = trial.suggest_categorical("sde_sample_freq", [-1, 8, 16, 32, 64])
    normalize_advantage = trial.suggest_categorical("normalize_advantage", [True, False])
    activation_fn = trial.suggest_categorical("activation_fn", ["tanh", "relu"])
    activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU}[activation_fn]

    return {
        "learning_rate": learning_rate,
        "n_steps": n_steps,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "ent_coef": ent_coef,
        "vf_coef": vf_coef,
        "max_grad_norm": max_grad_norm,
        "rms_prop_eps": rms_prop_eps,
        "use_rms_prop": use_rms_prop,
        "use_sde": False, # For discrete action env
        "sde_sample_freq": -1, 
        "normalize_advantage": normalize_advantage,
        "policy_kwargs": {
            "activation_fn": activation_fn,
        },
    }


def objective(trial: optuna.Trial):
    kwargs = sample_a2c_params(trial)
    train_env = DummyVecEnv([make_env(ENV_ID, i, "log/a2c") for i in range(NUM_ENV)])
    eval_env = DummyVecEnv([lambda: Monitor(gym.make(ENV_ID))])
    
    eval_callback = TrialEvalCallback(
        eval_env,
        trial,
        n_eval_episodes=10,
        eval_freq=EVAL_FREQ,
        deterministic=True,
    )

    model = sb3.A2C("MlpPolicy", train_env, verbose=0, **kwargs)

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

    study_name = f"a2c_{ENV_ID}_optimization"
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
    csv_path = os.path.join(output_dir, f"study_results_a2c_{ENV_ID}.csv")
    study.trials_dataframe().to_csv(csv_path, index=False)


def train_agent(log_dir = "log/a2c"):
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    train_env = DummyVecEnv([make_env(ENV_ID, i, log_dir) for i in range(NUM_ENV)])
    eval_env = DummyVecEnv([lambda: Monitor(gym.make(ENV_ID))])
    callback_on_thr = StopTrainingOnRewardThreshold(reward_threshold=500, verbose=1)
    # callback_on_best = StopTrainingOnNoModelImprovement(max_no_improvement_evals=2, min_evals=10, verbose=1)
    eval_callback = EvalCallback(eval_env, n_eval_episodes=50, eval_freq=EVAL_FREQ, callback_on_new_best=callback_on_thr, verbose=1)
    #eval_callback = EvalCallback(eval_env, n_eval_episodes=50, callback_after_eval=callback_on_best, verbose=1, eval_freq=1000)

    best_params = {
        "learning_rate": 0.0006670754981097518,
        "n_steps": 16,
        "gamma": 0.9748249781119694,
        "gae_lambda": 0.9349807642322937,
        "ent_coef": 1.3272886003999022e-05,
        "vf_coef": 0.56233812874183,
        "max_grad_norm": 1.7837557184548907,
        "rms_prop_eps": 5.20249107325663e-06,
        "use_rms_prop": False,
        "normalize_advantage": False,
        "policy_kwargs": {"activation_fn": nn.Tanh} # từ activation_fn: tanh
    }

    model = sb3.A2C(
        "MlpPolicy", 
        train_env, 
        verbose=1,
        learning_rate=best_params["learning_rate"],
        n_steps=best_params["n_steps"],
        gamma=best_params["gamma"],
        gae_lambda=best_params["gae_lambda"],
        ent_coef=best_params["ent_coef"],
        vf_coef=best_params["vf_coef"],
        max_grad_norm=best_params["max_grad_norm"],
        rms_prop_eps=best_params["rms_prop_eps"],
        use_rms_prop=best_params["use_rms_prop"],
        normalize_advantage=best_params["normalize_advantage"],
        policy_kwargs=best_params["policy_kwargs"],
    )

    model.learn(total_timesteps=TOTAL_TRAINING_STEPS, callback=eval_callback, log_interval=None, progress_bar=True)
    model.save(MODEL_PATH)


def test_agent():
    env = gym.make(ENV_ID, render_mode="human")
    model = sb3.A2C.load(MODEL_PATH, env)
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
    # visualize_optimized_results(f"a2c_{ENV_ID}_optimization")
    train_agent()
    # advanced_plot_training_result()
    # test_agent()


if __name__ == "__main__":
    main()
