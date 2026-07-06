import gymnasium as gym
import os
import torch
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize


def make_env(env_id, rank, log_dir):
    def _init():
        os.makedirs(log_dir, exist_ok=True)
        env = gym.make(env_id)
        monitor_log_path = os.path.join(log_dir, str(rank)) 
        env = Monitor(env, monitor_log_path)
        return env
    return _init   


def make_subproc_vec_env(env_id, n_env, log_dir):
    os.makedirs(log_dir, exist_ok=True)
    torch.set_num_threads(1)
    return SubprocVecEnv(
        [make_env(env_id, i, log_dir) for i in range(n_env)],
        start_method="forkserver",
    )


def make_normalized_vec_env(env_fns, training=True):
    return VecNormalize(
        DummyVecEnv(env_fns),
        norm_obs=True,
        norm_reward=False,
        training=training,
    )
