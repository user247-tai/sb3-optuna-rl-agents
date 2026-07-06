# DRL SB3 Workspace

This workspace is a small reinforcement-learning lab built around
[Gymnasium](https://gymnasium.farama.org/), [Stable-Baselines3](https://stable-baselines3.readthedocs.io/),
[sb3-contrib](https://sb3-contrib.readthedocs.io/), [PyTorch](https://pytorch.org/), and
[Optuna](https://optuna.readthedocs.io/).

<p align="center">
  <img width="240" height="240" alt="humanoid_standup_sac(1)" src="https://github.com/user-attachments/assets/2707bef3-2273-4022-9b6c-5ab6fd6149d5" />
</p>

It contains one training script per agent, hyperparameter search utilities, logging helpers,
saved models, and rollout visualizations.

## Overview

The repo focuses on training and tuning a few common DRL agents:

- `DQN` and `A2C` for `CartPole-v1`
- `PPO` for `HumanoidStandup-v5`
- `TD3` for `Ant-v5`
- `SAC` for `HalfCheetah-v5`
- `ARS` for `HalfCheetah-v4`

Each agent script follows the same general structure:

1. Define the environment and training constants.
2. Sample hyperparameters with Optuna.
3. Train an SB3 model with vectorized environments and monitor logs.
4. Save the best trained model into `trained_model/`.
5. Optionally test the model and generate a GIF rollout.

### Main folders

- `agents/` - training, optimization, and testing entrypoints for each algorithm
- `utils/` - environment helpers and plotting utilities
- `custom/` - custom Optuna/SB3 callback(s)
- `log/` - monitor CSV files from training runs
- `db/` - Optuna SQLite studies, CSV exports, and HTML plots
- `trained_model/` - saved `.zip` models and normalization files
- `*.gif` - sample rollout animations generated from trained policies

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/user247-tai/sb3-optuna-rl-agents.git
cd drl_sb3
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Upgrade packaging tools

```bash
python -m pip install --upgrade pip setuptools wheel
```

### 4. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 5. Verify the installation

You can quickly confirm that the main modules import correctly:

```bash
python3 -c "import agents.SAC, agents.TD3, agents.PPO, agents.A2C, agents.DQN, agents.ARS; print('imports ok')"
```

You can also confirm that the repository paths and saved models are visible:

```bash
python3 -c "import agents.SAC as sac; print(sac.MODEL_PATH)"
```

### Notes

- The project uses MuJoCo environments, so `gymnasium[mujoco]` must be installed and working.
- If you want to render `test_agent()` rollouts on a local machine, make sure your OpenGL/GLFW setup is available.
- On a headless server, the render-based test functions may fail unless you configure offscreen rendering or use a virtual display.
- If your system does not already have the required native libraries for MuJoCo/OpenGL, install those first using your OS package manager.
- A Python 3.10+ environment is recommended.

## How To Use

Each file in `agents/` exposes the same high-level actions:

- `run_optimization()` - Optuna hyperparameter search
- `train_agent()` - train the final policy with selected hyperparameters
- `test_agent()` - load a saved model and run a test rollout

The scripts also have a `main()` function at the bottom. In most files, one action is enabled
and the others are commented out. You can switch behavior by editing the file or by calling the
function directly from Python.

### 1. Optimize parameters

Example:

```bash
python3 -c "import agents.SAC as sac; sac.run_optimization()"
python3 -c "import agents.TD3 as td3; td3.run_optimization()"
python3 -c "import agents.PPO as ppo; ppo.run_optimization()"
python3 -c "import agents.A2C as a2c; a2c.run_optimization()"
python3 -c "import agents.DQN as dqn; dqn.run_optimization()"
python3 -c "import agents.ARS as ars; ars.run_optimization()"
```

What it does:

- creates or reuses an Optuna SQLite study in `db/`
- runs repeated training trials
- prunes bad trials early with `TrialEvalCallback`
- writes a CSV summary of the search results

### 2. Train the final agent

Example:

```bash
python3 -c "import agents.SAC as sac; sac.train_agent()"
python3 -c "import agents.TD3 as td3; td3.train_agent()"
python3 -c "import agents.PPO as ppo; ppo.train_agent()"
python3 -c "import agents.A2C as a2c; a2c.train_agent()"
python3 -c "import agents.DQN as dqn; dqn.train_agent()"
python3 -c "import agents.ARS as ars; ars.train_agent()"
```

Training outputs:

- saved models in `trained_model/`
- monitor logs in `log/<agent>/`
- evaluation checkpoints and best models when enabled

### 3. Test a trained model

The built-in `test_agent()` functions load a saved model and roll out one episode.
Some of them also render frames and save a GIF.

Examples:

```bash
python3 -c "import agents.SAC as sac; sac.test_agent()"
python3 -c "import agents.TD3 as td3; td3.test_agent()"
python3 -c "import agents.PPO as ppo; ppo.test_agent()"
python3 -c "import agents.ARS as ars; ars.test_agent()"
python3 -c "import agents.A2C as a2c; a2c.test_agent()"
python3 -c "import agents.DQN as dqn; dqn.test_agent()"
```

Important:

- The render-based tests may fail in a headless environment.
- `ARS` also relies on the saved `VecNormalize` statistics file.
- The comparison table below only uses the saved `.zip` files for `TD3`, `PPO`, and `SAC`.

## Saved Model Comparison

The table below compares the saved `.zip` models for `TD3`, `PPO`, and `SAC` on the three
MuJoCo environments that are present in this workspace: `Ant-v5`, `HalfCheetah-v5`, and
`HumanoidStandup-v5`.

These are deterministic single-episode returns captured from the saved models in `trained_model/`.
They are useful as a snapshot, but they are **not directly apples-to-apples** because the reward
scales differ across environments.

| Environment | TD3 | PPO | SAC |
| --- | ---: | ---: | ---: |
| `Ant-v5` | `2443.83` | `-2.17` | `5031.41` |
| `HalfCheetah-v5` | `2612.18` | `1567.50` | `14216.32` |
| `HumanoidStandup-v5` | `166959.16` | `99571.09` | `273421.99` |

## Generated Artifacts

Running optimization and training can create or update:

- `db/*.db` - Optuna studies
- `db/*.csv` - search result summaries
- `db/*.html` - interactive Optuna visualizations
- `log/*/*.monitor.csv` - episode logs
- `trained_model/*.zip` - SB3 model checkpoints
- `trained_model/*vecnormalize.pkl` - normalization statistics for vectorized envs
- `*.gif` - rollout visualizations

## Notes

- The workspace is more of an experiment harness than a reusable library package.
- Most scripts set `PROJECT_ROOT` and change the working directory so they can be run from anywhere.
- If you want a fair benchmark between agents, evaluate them on the same environment, same seed set, and same number of episodes.
