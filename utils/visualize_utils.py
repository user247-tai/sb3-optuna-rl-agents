import os
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import optuna
from optuna.visualization import *
from stable_baselines3.common import results_plotter
from stable_baselines3.common.results_plotter import plot_results
from stable_baselines3.common.monitor import load_results
from stable_baselines3.common.results_plotter import ts2xy, window_func

def visualize_optimized_results(db_name="optimization_result"):
    # Create the path to the 'db' subdirectory
    output_dir = "db"
    
    # Ensure the directory exists to avoid FileNotFoundError
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # Configure the database storage path inside the 'db' folder
    study_name = db_name
    storage_name = f"sqlite:///{output_dir}/{study_name}.db"

    try:
        # Load the study from the SQLite database file
        study = optuna.load_study(study_name=study_name, storage=storage_name)
        
        print(f"Successfully loaded Study: {study_name}")
        print(f"Total number of Trials found: {len(study.trials)}")
        print(f"Best objective value: {study.best_value}")
        
        # Check if there is enough data to generate plots (requires at least 2 trials)
        if len(study.trials) > 1:
            # Save interactive HTML plots into the 'db' directory
            
            # 1. Optimization History: Shows the improvement of the objective value over trials
            fig1 = plot_optimization_history(study)
            fig1.write_html(os.path.join(output_dir, "optimization_history.html"))

            # 2. Hyperparameter Importances: Identifies which parameters affect performance the most
            fig2 = plot_param_importances(study)
            fig2.write_html(os.path.join(output_dir, "param_importances.html"))

            # 3. Parallel Coordinate: Visualizes high-dimensional relationships between parameters
            fig3 = plot_parallel_coordinate(study)
            fig3.write_html(os.path.join(output_dir, "parallel_coordinate.html"))

            # 4. Slice Plot: Shows the relationship between individual parameters and objective value
            fig4 = plot_slice(study)
            fig4.write_html(os.path.join(output_dir, "slice.html"))
            
            print(f"All visualization files have been saved to: {output_dir}/")

        else:
            print("Not enough data (at least 2 completed trials required) to visualize.")

    except KeyError:
        print(f"Error: Study name '{study_name}' not found in {storage_name}")
    except Exception as e:
        print(f"An error occurred while loading data: {e}")

def advanced_plot_training_result(log_dir="log/"):
    # Load the results
    df = load_results(log_dir)

    # Convert dataframe (x=timesteps, y=episodic return)
    x, y = ts2xy(df, "timesteps")

    # Plot raw data
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.scatter(x, y, s=2, alpha=0.6)
    plt.xlabel("Timesteps")
    plt.ylabel("Episode Reward")
    plt.title("Raw Episode Rewards")

    # Plot smoothed data with custom window
    plt.subplot(2, 1, 2)
    if len(x) >= 50:  # Only smooth if we have enough data
        x_smooth, y_smooth = window_func(x, y, 50, np.mean)
        plt.plot(x_smooth, y_smooth, linewidth=2)
        plt.xlabel("Timesteps")
        plt.ylabel("Average Episode Reward (50-episode window)")
        plt.title("Smoothed Episode Rewards")

    plt.tight_layout()
    plt.show()

def plot_training_result(log_dir="log/"):
    plot_results([log_dir], 500_000, results_plotter.X_EPISODES, "DQN CartPole")
    plt.tight_layout()
    plt.show()