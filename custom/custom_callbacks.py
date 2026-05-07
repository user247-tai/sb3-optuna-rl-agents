from stable_baselines3.common.callbacks import *

class TrialEvalCallback(EvalCallback):
    def __init__(self, eval_env, trial, n_eval_episodes=5, eval_freq=10000, deterministic=True, verbose=0):
        super().__init__(eval_env=eval_env, 
                        n_eval_episodes=n_eval_episodes, 
                        eval_freq=eval_freq, 
                        deterministic=deterministic, 
                        verbose=verbose)
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False
        self.last_step_count = 0

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and (self.n_calls - self.last_step_count) >= self.eval_freq:
            self.last_step_count = self.n_calls 
            
            super()._on_step()
            
            self.eval_idx += 1
            self.trial.report(self.last_mean_reward, self.eval_idx)
            
            if self.trial.should_prune():
                self.is_pruned = True
                return False
        return True