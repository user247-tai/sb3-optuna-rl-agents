import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import gymnasium as gym
import stable_baselines3 as sb3
from stable_baselines3.common.callbacks import *
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import torch
import torch.nn as nn
from custom.custom_callbacks import TrialEvalCallback
from utils.env_utils import make_env, make_subproc_vec_env
from utils.visualize_utils import visualize_optimized_results, advanced_plot_training_result, plot_training_result

NUM_ENV = 15
ENV_ID = "HumanoidStandup-v5"
EVAL_FREQ = max(200_000 // NUM_ENV, 1)
TOTAL_OPTIMIZE_STEPS = 1_000_000
TOTAL_TRAINING_STEPS = 10_000_000
MODEL_DIR = "trained_model"
MODEL_PATH = os.path.join(MODEL_DIR, f"ppo_{ENV_ID}")


def sample_ppo_params(trial: optuna.Trial):    
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)    
    n_steps = trial.suggest_categorical("n_steps", [128, 256, 512, 1024, 2048])    
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])    
    n_epochs = trial.suggest_int("n_epochs", 3, 20)    
    gamma = trial.suggest_float("gamma", 0.95, 0.9999, log=True)    
    gae_lambda = trial.suggest_float("gae_lambda", 0.8, 1.0)    
    clip_range = trial.suggest_float("clip_range", 0.1, 0.4)
    clip_range_vf = trial.suggest_categorical("clip_range_vf", [None, 0.5, 1.0]) # IMPORTANT: this clipping depends on the reward scaling.
    normalize_advantage = trial.suggest_categorical("normalize_advantage", [True, False])
    ent_coef = trial.suggest_float("ent_coef", 1e-8, 0.01, log=True)
    vf_coef = trial.suggest_float("vf_coef", 0.1, 0.9)
    max_grad_norm = trial.suggest_float("max_grad_norm", 0.3, 5.0)
    use_sde = trial.suggest_categorical("use_sde", [True, False])
    sde_sample_freq = -1
    if use_sde:
        sde_sample_freq = trial.suggest_categorical("sde_sample_freq", [8, 16, 32, 64])
    else:
        trial.set_user_attr("sde_sample_freq", -1)

    target_kl = trial.suggest_float("target_kl", 0.003, 0.03, log=True)
    activation_fn_name = trial.suggest_categorical("activation_fn", ["tanh", "relu"])
    activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU}[activation_fn_name]
    ortho_init = trial.suggest_categorical("ortho_init", [True, False])

    return {
        "learning_rate": learning_rate,
        "n_steps": n_steps,
        "batch_size": batch_size,
        "n_epochs": n_epochs,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "clip_range": clip_range,
        "clip_range_vf": clip_range_vf,
        "normalize_advantage": normalize_advantage,
        "ent_coef": ent_coef,
        "vf_coef": vf_coef,
        "max_grad_norm": max_grad_norm,
        "use_sde": use_sde,
        "sde_sample_freq": sde_sample_freq,
        "target_kl": target_kl,
        "policy_kwargs": {
            "activation_fn": activation_fn,
            "ortho_init": ortho_init,
            "log_std_init": trial.suggest_float("log_std_init", -4, -1),
            "net_arch": [256, 256], # Standard MLP architecture for MuJoCo environments.
        },
        "device": "cpu",
    }


def objective(trial: optuna.Trial):
    kwargs = sample_ppo_params(trial)
    train_env = make_subproc_vec_env(ENV_ID, NUM_ENV, "log/ppo")
    eval_env = DummyVecEnv([lambda: Monitor(gym.make(ENV_ID))])
    
    eval_callback = TrialEvalCallback(
        eval_env,
        trial,
        n_eval_episodes=10,
        eval_freq=EVAL_FREQ,
        deterministic=True,
    )

    model = sb3.PPO("MlpPolicy", train_env, verbose=0, **kwargs)

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

    study_name = f"ppo_{ENV_ID}_optimization"
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
        study.optimize(objective, n_trials=25, timeout=None)
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
    csv_path = os.path.join(output_dir, f"study_results_ppo_{ENV_ID}.csv")
    study.trials_dataframe().to_csv(csv_path, index=False)


def train_agent(log_dir = "log/ppo"):
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    train_env = make_subproc_vec_env(ENV_ID, NUM_ENV, log_dir)
    eval_env = DummyVecEnv([lambda: Monitor(gym.make(ENV_ID))])
    # callback_on_thr = StopTrainingOnRewardThreshold(reward_threshold=500, verbose=1)
    # callback_on_best = StopTrainingOnNoModelImprovement(max_no_improvement_evals=2, min_evals=10, verbose=1)
    eval_callback = EvalCallback(eval_env, n_eval_episodes=10, eval_freq=EVAL_FREQ, verbose=1)
    #eval_callback = EvalCallback(eval_env, n_eval_episodes=50, callback_after_eval=callback_on_best, verbose=1, eval_freq=1000)

    policy_kwargs = {
        "activation_fn": nn.Tanh,        
        "ortho_init": True,              
        "log_std_init": -1.5447685018374857,
        "net_arch": [256, 256], 
    }

    best_params = {
        "learning_rate": 0.00014959809387471836,
        "n_steps": 128,
        "batch_size": 128 * NUM_ENV, #512,
        "n_epochs": 15,
        "gamma": 0.959537343272522,
        "gae_lambda": 0.8973236970919566,
        "clip_range": 0.2826452974146355,
        "clip_range_vf": 1.0,
        "normalize_advantage": True,
        "ent_coef": 0.005667770103497358,
        "vf_coef": 0.21090334186540152,
        "max_grad_norm": 4.817690670268407,
        "use_sde": True,
        "sde_sample_freq": 64,
        "target_kl": 0.02,
        "policy_kwargs": policy_kwargs,
    }

    model = sb3.PPO(
        "MlpPolicy", 
        train_env, 
        verbose=1,
        learning_rate=best_params["learning_rate"],
        n_steps=best_params["n_steps"],
        batch_size=best_params["batch_size"],
        n_epochs=best_params["n_epochs"],
        gamma=best_params["gamma"],
        gae_lambda=best_params["gae_lambda"],
        clip_range=best_params["clip_range"],
        clip_range_vf=best_params["clip_range_vf"],
        normalize_advantage=best_params["normalize_advantage"],
        ent_coef=best_params["ent_coef"],
        vf_coef=best_params["vf_coef"],
        max_grad_norm=best_params["max_grad_norm"],
        use_sde=best_params["use_sde"],
        sde_sample_freq=best_params["sde_sample_freq"],
        target_kl=best_params["target_kl"],
        policy_kwargs=best_params["policy_kwargs"],
        device="cpu"
    )

    model.learn(total_timesteps=TOTAL_TRAINING_STEPS, callback=eval_callback, log_interval=None, progress_bar=True)
    model.save(MODEL_PATH)


def test_agent():
    env = gym.make(ENV_ID, render_mode="human")
    model = sb3.PPO.load(MODEL_PATH, env)
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
    # visualize_optimized_results(f"ppo_{ENV_ID}_optimization")
    train_agent()
    # advanced_plot_training_result()
    # test_cartpole_agent()


if __name__ == "__main__":
    main()
