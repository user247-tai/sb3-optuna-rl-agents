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
EVAL_FREQ = max(5_000 // NUM_ENV, 1)
TOTAL_OPTIMIZE_STEPS = 100_000
TOTAL_TRAINING_STEPS = 500_000
MODEL_DIR = "trained_model"
MODEL_PATH = os.path.join(MODEL_DIR, f"dqn_{ENV_ID}")


def sample_dqn_params(trial: optuna.Trial):
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    buffer_size_pow = trial.suggest_int("buffer_size_pow", 2, 7)
    buffer_size = 10 ** buffer_size_pow
    learning_starts_pow = trial.suggest_int("learning_starts_pow", 2, 4)
    learning_starts = 10 ** learning_starts_pow
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256, 512])
    tau = trial.suggest_float("tau", 0.001, 0.1, log=True)
    gamma = trial.suggest_float("gamma", 0.9, 0.999, log=True)
    train_freq_div = trial.suggest_int("train_freq_div", 1, 10)
    train_freq = train_freq_div * NUM_ENV
    n_steps = trial.suggest_int("n_steps", 1, 3)
    exploration_fraction = trial.suggest_float("exploration_fraction", 0.001, 0.15, log=True)
    exploration_initial_eps = trial.suggest_float("exploration_initial_eps", 0.8, 1.0)
    exploration_final_eps = trial.suggest_float("exploration_final_eps", 0.01, 0.1, log=True)
    activation_fn = trial.suggest_categorical("activation_fn", ["tanh", "relu"])
    activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU}[activation_fn]

    return {
        "learning_rate": learning_rate,
        "buffer_size": buffer_size,
        "learning_starts": learning_starts,
        "batch_size": batch_size,
        "tau": tau,
        "gamma": gamma,
        "train_freq": train_freq,
        "gradient_steps": -1,
        "n_steps": n_steps,
        "target_update_interval": 1, # For soft-update
        "exploration_fraction": exploration_fraction,
        "exploration_initial_eps": exploration_initial_eps,
        "exploration_final_eps": exploration_final_eps,
        "policy_kwargs": {
            "activation_fn": activation_fn,
        },
    }


def objective(trial: optuna.Trial):
    kwargs = sample_dqn_params(trial)
    train_env = DummyVecEnv([make_env(ENV_ID, i, "log/dqn") for i in range(NUM_ENV)])
    eval_env = DummyVecEnv([lambda: Monitor(gym.make(ENV_ID))])
    
    eval_callback = TrialEvalCallback(
        eval_env,
        trial,
        n_eval_episodes=10,
        eval_freq=EVAL_FREQ,
        deterministic=True,
    )

    model = sb3.DQN("MlpPolicy", train_env, verbose=0, **kwargs)

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


def train_agent(log_dir = "log/dqn"):
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    train_env = DummyVecEnv([make_env(ENV_ID, i, log_dir) for i in range(NUM_ENV)])
    eval_env = DummyVecEnv([lambda: Monitor(gym.make(ENV_ID))])
    callback_on_thr = StopTrainingOnRewardThreshold(reward_threshold=500, verbose=1)
    # callback_on_best = StopTrainingOnNoModelImprovement(max_no_improvement_evals=2, min_evals=10, verbose=1)
    eval_callback = EvalCallback(eval_env, n_eval_episodes=50, eval_freq=EVAL_FREQ, callback_on_new_best=callback_on_thr, verbose=1)
    #eval_callback = EvalCallback(eval_env, n_eval_episodes=50, callback_after_eval=callback_on_best, verbose=1, eval_freq=1000)

    best_params = {
        "learning_rate": 0.0005500280119818322,
        "buffer_size": 10**6,                # buffer_size_pow: 6 (1 triệu mẫu)
        "learning_starts": 10**4,            # learning_starts_pow: 4 (10,000 mẫu)
        "batch_size": 64,
        "tau": 0.026391395392824306,
        "gamma": 0.9982378452515824,         # Rất gần 1.0
        "train_freq": 6 * 10,                # train_freq_div (6) * NUM_ENV (10) = 60
        "gradient_steps": -1,
        "target_update_interval": 10,
        "exploration_fraction": 0.04512434259095921,
        "exploration_initial_eps": 0.9018606536628612,
        "exploration_final_eps": 0.010672952291781838,
        "max_grad_norm": 1.0156251712854574,
        "policy_kwargs": {"activation_fn": nn.Tanh} # activation_fn: tanh
    }

    model = sb3.DQN(
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
        target_update_interval=best_params["target_update_interval"],
        exploration_fraction=best_params["exploration_fraction"],
        exploration_initial_eps=best_params["exploration_initial_eps"],
        exploration_final_eps=best_params["exploration_final_eps"],
        max_grad_norm=best_params["max_grad_norm"],
        policy_kwargs=best_params["policy_kwargs"]
    )

    model.learn(total_timesteps=TOTAL_TRAINING_STEPS, callback=eval_callback, log_interval=None, progress_bar=True)
    model.save(MODEL_PATH)


def test_agent():
    env = gym.make(ENV_ID, render_mode="human")
    model = sb3.DQN.load(MODEL_PATH, env)
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


def run_optimization():
    # Create the path to the 'db' subdirectory
    output_dir = "db"
    
    # Ensure the directory exists to avoid FileNotFoundError
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    study_name = f"dqn_{ENV_ID}_optimization"
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
        study.optimize(objective, n_trials=30, timeout=None)
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
    study.trials_dataframe().to_csv(f"study_results_dqn_{ENV_ID}.csv")


def main():
    # run_optimization()
    # visualize_optimized_results(f"dqn_{ENV_ID}_optimization")
    train_agent()
    # advanced_plot_training_result()
    # test_agent()


if __name__ == "__main__":
    main()
