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
MODEL_PATH = os.path.join(MODEL_DIR, f"td3_{ENV_ID}")

def sample_td3_params(trial: optuna.Trial, n_actions: int):
    """Sampler for TD3 hyperparameters."""
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 7e-4, log=True)
    buffer_size_pow = trial.suggest_int("buffer_size_pow", 2, 7)
    buffer_size = 10 ** buffer_size_pow
    learning_starts_pow = trial.suggest_int("learning_starts_pow", 2, 4)
    learning_starts = 10 ** learning_starts_pow
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256, 512])
    tau = trial.suggest_float("tau", 0.001, 0.1, log=True)
    gamma = trial.suggest_float("gamma", 0.9, 0.9999, log=True)
    train_freq_div = trial.suggest_int("train_freq_div", 1, 8)
    train_freq = train_freq_div * NUM_ENV
    gradient_steps = trial.suggest_int("gradient_steps", 1, NUM_ENV)
    n_steps = trial.suggest_int("n_steps", 1, 3)
    policy_delay = trial.suggest_int("policy_delay", 1, 5)
    target_policy_noise = trial.suggest_float("target_policy_noise", 0.1, 0.3) # Tune base on env
    target_noise_clip = trial.suggest_float("target_noise_clip", 0.2, 0.75) # Tune base on env
    activation_fn = trial.suggest_categorical("activation_fn", ["tanh", "relu"])
    activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU}[activation_fn]
    noise_type = trial.suggest_categorical("action_noise", ["normal", "ornstein"])
    noise_sigma = trial.suggest_float("noise_sigma", 0.1, 0.3)
    
    if noise_type == "normal":
        base_noise = sb3_noise.NormalActionNoise(
            mean=np.zeros(n_actions), 
            sigma=noise_sigma * np.ones(n_actions)
        )
    else:
        base_noise = sb3_noise.OrnsteinUhlenbeckActionNoise(
            mean=np.zeros(n_actions), 
            sigma=noise_sigma * np.ones(n_actions)
        )
        
    action_noise_vec = sb3_noise.VectorizedActionNoise(base_noise=base_noise, n_envs=NUM_ENV)

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
        "policy_delay": policy_delay,
        "target_policy_noise": target_policy_noise, # Tune base on env
        "target_noise_clip": target_noise_clip, # Tune base on env
        "policy_kwargs": {
            "activation_fn": activation_fn,
        },
        "action_noise": action_noise_vec,
    }


def objective(trial: optuna.Trial):
    train_env = DummyVecEnv([make_env(ENV_ID, i, "log/td3") for i in range(NUM_ENV)])
    eval_env = DummyVecEnv([lambda: Monitor(gym.make(ENV_ID))])
    kwargs = sample_td3_params(trial, train_env.action_space.shape[0])
    
    eval_callback = TrialEvalCallback(
        eval_env,
        trial,
        n_eval_episodes=50,
        eval_freq=EVAL_FREQ,
        deterministic=True,
    )

    model = sb3.TD3("MlpPolicy", train_env, verbose=0, **kwargs)

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

    study_name = f"td3_{ENV_ID}_optimization"
    storage_name = f"sqlite:///{output_dir}/{study_name}.db"
    sampler = TPESampler(n_startup_trials=15, multivariate=True)

    pruner = MedianPruner(n_startup_trials=15, n_warmup_steps=2)
    
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
    csv_path = os.path.join(output_dir, f"study_results_td3_{ENV_ID}.csv")
    study.trials_dataframe().to_csv(csv_path, index=False)


def train_agent(log_dir = "log/td3"):
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    train_env = DummyVecEnv([make_env(ENV_ID, i, log_dir) for i in range(NUM_ENV)])
    eval_env = DummyVecEnv([lambda: Monitor(gym.make(ENV_ID))])
    # callback_on_thr = StopTrainingOnRewardThreshold(reward_threshold=500, verbose=1)
    # callback_on_best = StopTrainingOnNoModelImprovement(max_no_improvement_evals=2, min_evals=10, verbose=1)
    eval_callback = EvalCallback(eval_env, n_eval_episodes=50, eval_freq=EVAL_FREQ, verbose=1)
    #eval_callback = EvalCallback(eval_env, n_eval_episodes=50, callback_after_eval=callback_on_best, verbose=1, eval_freq=1000)

    buffer_size = int(10**5)
    learning_starts = int(10**4)
    n_actions = train_env.action_space.shape[-1]
    action_noise = sb3_noise.NormalActionNoise(mean=np.zeros(n_actions), sigma=np.full(n_actions, 0.24815138090207658))
    action_noise_vec = sb3_noise.VectorizedActionNoise(base_noise=action_noise, n_envs=NUM_ENV)

    best_params = {
        "learning_rate": 4.713128617246661e-05,
        "buffer_size": buffer_size,
        "learning_starts": learning_starts,
        "batch_size": 128,
        "tau": 0.013883517929503198,
        "gamma": 0.9008559697177815,
        "train_freq": 1, 
        "gradient_steps": 9,
        "policy_delay": 3,
        "target_policy_noise": 0.20833810267000297,
        "target_noise_clip": 0.2420689169979912,
        "policy_kwargs": {"activation_fn": nn.Tanh}, 
        "action_noise": action_noise_vec
    }

    model = sb3.TD3(
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
        policy_delay=best_params["policy_delay"],
        target_policy_noise=best_params["target_policy_noise"],
        target_noise_clip=best_params["target_noise_clip"],
        policy_kwargs=best_params["policy_kwargs"],
        action_noise=best_params["action_noise"],
    )

    model.learn(total_timesteps=TOTAL_TRAINING_STEPS, callback=eval_callback, log_interval=None, progress_bar=True)
    model.save(MODEL_PATH)


def test_agent():
    env = gym.make(ENV_ID, render_mode="human")
    model = sb3.TD3.load(MODEL_PATH, env)
    stop = False
    obs, _ = env.reset()
    eps_return = 0
    while not stop:
        action, _ = model.predict(observation=obs)
        next_obs, reward, terminate, truncate, info = env.step(action)
        eps_return += reward
        obs = next_obs
        stop = terminate or truncate

    env.close()
    print(f"Episode return: {eps_return}")


def main():
    # run_optimization()
    # visualize_optimized_results(f"td3_{ENV_ID}_optimization")
    train_agent()
    # advanced_plot_training_result()
    # test_agent()


if __name__ == "__main__":
    main()
