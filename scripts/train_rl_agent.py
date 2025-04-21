import os
import sys # <-- Import sys
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
# Import PettingZoo wrapper if needed directly, but make_vec_env might handle it
# from stable_baselines3.common.vec_env import VecPettingZoo

# --- Add project root to sys.path ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)
# --- End of added code ---

# Import your custom environment constructor
from environment.negotiation_env import env as negotiation_env, MAX_ITEM_QUANTITY, NUM_ITEMS

# --- Configuration ---
TRAIN_ALGORITHM = PPO
POLICY_TYPE = "MlpPolicy" # Use "MultiInputPolicy" if dealing with Dict spaces and MlpPolicy fails
TOTAL_TIMESTEPS = 100_000 # Adjust as needed (start small for testing)
NUM_ENVIRONMENTS = 4 # Number of parallel environments for faster training (adjust based on CPU cores)
LEARNING_RATE = 3e-4 # Typical learning rate for PPO
BATCH_SIZE = 64 # PPO default
N_STEPS = 2048 # PPO default (steps per environment per update)

MODEL_SAVE_DIR = "models"
MODEL_SAVE_NAME = "ppo_negotiator"
LOG_DIR = "logs/ppo_negotiator_logs"

# Ensure directories exist
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

def create_env_lambda(env_config):
    """Lambda function to create an instance of the environment."""
    # Important: Define utility functions here or ensure they are accessible
    # For simplicity, using the default utilities from the env definition
    env = negotiation_env(**env_config)
    return env

if __name__ == "__main__":
    print("--- Starting RL Agent Training ---")

    # --- Environment Setup ---
    # Define utility functions for the training environments
    # Example: Symmetric utilities for training a general negotiator
    agent_utilities = {
         "agent_0": lambda offer: offer[0] * 0.5 + offer[1] * 0.5,
         "agent_1": lambda offer: offer[0] * 0.5 + offer[1] * 0.5,
    }
    # Or use asymmetric ones if training for a specific role
    # agent_utilities = {
    #     "agent_0": lambda offer: offer[0] * 0.8 + offer[1] * 0.2,
    #     "agent_1": lambda offer: offer[0] * 0.1 + offer[1] * 0.9,
    # }

    env_config = {
        "max_rounds": 20,
        "render_mode": None, # No rendering during training
        "agent_utils": agent_utilities
    }

    # --- Vectorized Environment ---
    # Create a function that returns a function to create the environment
    # This is needed for parallel environments
    env_fn = lambda: create_env_lambda(env_config)

    # Use make_vec_env which handles PettingZoo conversion via pettingzoo_env_to_vec_env
    # It also automatically wraps with VecMonitor for logging
    print(f"Creating {NUM_ENVIRONMENTS} parallel environments...")
    vec_env = make_vec_env(
        env_fn,
        n_envs=NUM_ENVIRONMENTS,
        vec_env_cls=SubprocVecEnv, # Use SubprocVecEnv for parallelism
        # Removed VecMonitor wrapper as make_vec_env adds it
        # vec_env_kwargs={'start_method': 'fork'} # Use 'fork' on macOS/Linux if needed, 'spawn' is safer/default
    )
    print("Vectorized environment created.")

    # --- Model Definition ---
    print(f"Initializing {TRAIN_ALGORITHM.__name__} model with {POLICY_TYPE}...")
    model = TRAIN_ALGORITHM(
        POLICY_TYPE,
        vec_env,
        verbose=1, # Print training progress
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        n_steps=N_STEPS,
        tensorboard_log=LOG_DIR,
        # Add other hyperparameters as needed (e.g., gamma, gae_lambda, n_epochs)
    )
    print("Model initialized.")

    # --- Training ---
    print(f"Starting training for {TOTAL_TIMESTEPS} timesteps...")
    # Callbacks can be added here for saving best models, etc.
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        log_interval=1, # Log every update
        tb_log_name=MODEL_SAVE_NAME, # Log name for TensorBoard
        reset_num_timesteps=False # Continue training if script is re-run with loaded model
    )
    print("Training finished.")

    # --- Save Model ---
    save_path = os.path.join(MODEL_SAVE_DIR, f"{MODEL_SAVE_NAME}_{TOTAL_TIMESTEPS}.zip")
    model.save(save_path)
    print(f"Model saved to {save_path}")

    # --- Cleanup ---
    vec_env.close()
    print("Environment closed.")
    print("--- Training Script Complete ---")

    print("\nTo monitor training with TensorBoard, run:")
    print(f"tensorboard --logdir {LOG_DIR}")