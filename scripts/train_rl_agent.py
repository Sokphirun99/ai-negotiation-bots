import os
import sys
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor, DummyVecEnv
import gymnasium
from gymnasium.spaces import Box, Discrete, Dict
import traceback  # Add this for better error reporting

# --- Add project root to sys.path ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)
# --- End of added code ---

from environment.negotiation_env import env as negotiation_env, MAX_ITEM_QUANTITY, NUM_ITEMS

# --- Custom PettingZoo to Gymnasium Wrapper ---
class PettingZooGymWrapper(gymnasium.Env):
    """Converts a PettingZoo AEC environment to a standard Gymnasium environment for a single agent."""
    
    def __init__(self, env, agent_id="agent_0"):
        """
        env: PettingZoo AEC environment
        agent_id: ID of the agent to train (default: "agent_0")
        """
        self.env = env
        self.agent_id = agent_id
        self.possible_agents = self.env.possible_agents
        
        # Reset the environment to get the action and observation spaces
        self.env.reset()
        
        # Store the original action and observation spaces
        self.original_action_space = self.env.action_space(self.agent_id)
        self.original_observation_space = self.env.observation_space(self.agent_id)
        
        # --- Convert Dict observation space to flat Box observation space ---
        if isinstance(self.original_observation_space, Dict):
            # Extract components and determine total size
            total_size = 0
            obs_low = []
            obs_high = []
            
            # Handle 'last_offer_received': Box(0, 10, (2,), int32)
            if "last_offer_received" in self.original_observation_space.spaces:
                offer_space = self.original_observation_space["last_offer_received"]
                total_size += np.prod(offer_space.shape)
                obs_low.extend([0] * np.prod(offer_space.shape))
                obs_high.extend([MAX_ITEM_QUANTITY] * np.prod(offer_space.shape))
            
            # Handle 'offer_valid': Discrete(2)
            if "offer_valid" in self.original_observation_space.spaces:
                total_size += 1
                obs_low.append(0)
                obs_high.append(1)
            
            # Handle 'round': Box(0, 20, (1,), int32)
            if "round" in self.original_observation_space.spaces:
                round_space = self.original_observation_space["round"]
                total_size += np.prod(round_space.shape)
                obs_low.extend([0] * np.prod(round_space.shape))
                obs_high.extend([20] * np.prod(round_space.shape))  # Assuming max_rounds=20
            
            # Create the flattened observation space
            self.observation_space = Box(
                low=np.array(obs_low, dtype=np.float32),
                high=np.array(obs_high, dtype=np.float32),
                dtype=np.float32
            )
            
            print(f"Converted observation space from {self.original_observation_space} to {self.observation_space}")
        else:
            # If not a Dict, just use the original
            self.observation_space = self.original_observation_space
        
        # --- Convert Dict action space to flat action space ---
        if isinstance(self.original_action_space, Dict):
            # Action type (accept=0, offer=1)
            self.action_type_space = self.original_action_space["type"]
            # Offer values (array of item quantities)
            self.offer_space = self.original_action_space["offer"]
            
            # Create a unified Box action space:
            # First value (index 0): action type (0=accept, 1=offer)
            # Remaining values (indices 1+): offer quantities for items
            low = np.array([0] + [0] * self.offer_space.shape[0], dtype=np.float32)
            high = np.array([1] + [MAX_ITEM_QUANTITY] * self.offer_space.shape[0], dtype=np.float32)
            self.action_space = Box(low=low, high=high, dtype=np.float32)
            
            print(f"Converted action space from {self.original_action_space} to {self.action_space}")
        else:
            # If not a Dict, just use the original
            self.action_space = self.original_action_space
    
    def _dict_to_flat_obs(self, dict_obs):
        """Convert a Dict observation to a flat Box observation."""
        flat_parts = []
        
        # Extract and flatten each component in the same order as in __init__
        if "last_offer_received" in dict_obs:
            flat_parts.append(dict_obs["last_offer_received"].flatten())
        
        if "offer_valid" in dict_obs:
            flat_parts.append(np.array([dict_obs["offer_valid"]], dtype=np.float32))
        
        if "round" in dict_obs:
            flat_parts.append(dict_obs["round"].flatten())
        
        # Concatenate all parts into a single flat array
        return np.concatenate(flat_parts).astype(np.float32)
    
    def reset(self, **kwargs):
        """Reset the environment and return the initial observation for our agent."""
        self.env.reset()
        
        # Skip to our agent's turn if it's not the first agent to act
        while self.env.agent_selection != self.agent_id and not self.env.terminations[self.agent_id]:
            if self.env.terminations[self.env.agent_selection] or self.env.truncations[self.env.agent_selection]:
                action = None
            else:
                # Take a random action for other agents
                action = self.env.action_space(self.env.agent_selection).sample()
            self.env.step(action)
        
        # Get the original observation
        original_obs = self.env.observe(self.agent_id)
        
        # Convert Dict observation to flat Box observation if needed
        if isinstance(self.original_observation_space, Dict):
            flat_obs = self._dict_to_flat_obs(original_obs)
        else:
            flat_obs = original_obs
        
        return flat_obs, {}  # Adding empty info dict for Gymnasium compatibility
    
    def step(self, action):
        """Take a step in the environment using the action for our agent."""
        try:
            # --- Convert flat action back to Dict format ---
            if isinstance(self.original_action_space, Dict):
                # Convert from Box to Dict
                # Extract action type (first value, rounded to int 0 or 1)
                action_type = int(np.round(action[0]))
                action_type = np.clip(action_type, 0, 1)  # Ensure it's 0 or 1
                
                # If accept action (type=0), no need for offer details
                if action_type == 0:
                    env_action = {"type": 0}  # Accept action doesn't need "offer" key
                # If offer action (type=1), include the offer values
                else:
                    # Extract offer values (remaining values, rounded to ints)
                    offer_values = np.round(action[1:]).astype(np.int32)
                    # Clip to valid range
                    offer_values = np.clip(offer_values, 0, MAX_ITEM_QUANTITY)
                    env_action = {"type": 1, "offer": offer_values}
            else:
                # If not using Dict conversion, pass through unchanged
                env_action = action
            
            # Take our agent's action in the original format the env expects
            self.env.step(env_action)
            
            # Have other agents act until it's our turn again
            done = False
            while self.env.agent_selection != self.agent_id and not done:
                if self.env.agent_selection not in self.env.agents:
                    # Environment might have reset between steps
                    break
                    
                if self.env.terminations.get(self.env.agent_selection, False) or self.env.truncations.get(self.env.agent_selection, False):
                    self.env.step(None)  # Terminated agents must pass None
                else:
                    # Take a random action for other agents
                    other_action = self.env.action_space(self.env.agent_selection).sample()
                    self.env.step(other_action)
                
                # Check if the environment is done after other agents act
                done = all(self.env.terminations.values()) or all(self.env.truncations.values())
            
            # Get the next observation, reward, termination flag, and truncation flag for our agent
            # Check if agent still exists (might have been removed if env reset)
            if self.agent_id in self.env.agents:
                reward = self.env.rewards.get(self.agent_id, 0.0)
                terminated = self.env.terminations.get(self.agent_id, False)
                truncated = self.env.truncations.get(self.agent_id, False)
                
                if not done:
                    original_next_obs = self.env.observe(self.agent_id)
                    # Convert Dict observation to flat Box observation if needed
                    if isinstance(self.original_observation_space, Dict):
                        next_obs = self._dict_to_flat_obs(original_next_obs)
                    else:
                        next_obs = original_next_obs
                else:
                    # For terminal states, return zeros observation
                    next_obs = np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype)
            else:
                # Agent no longer exists (environment likely reset)
                reward = 0.0
                terminated = True
                truncated = False
                # Return zeros observation for terminal state
                next_obs = np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype)
            
            # Additional info for debugging
            info = {"other_agents": [agent for agent in self.env.agents if agent != self.agent_id]}
            
            # If environment is done but our agent wasn't terminated or truncated,
            # we need to indicate termination for compatibility with SB3
            if done and not (terminated or truncated):
                terminated = True
            
            return next_obs, reward, terminated, truncated, info
            
        except Exception as e:
            print(f"Error in step method: {e}")
            print(traceback.format_exc())
            # Return a safe fallback response
            zero_obs = np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype)
            return zero_obs, 0.0, True, False, {}
    
    def close(self):
        """Close the environment."""
        self.env.close()

# --- Configuration ---
TRAIN_ALGORITHM = PPO
POLICY_TYPE = "MlpPolicy"
TOTAL_TIMESTEPS = 10_000  # Reduce for faster testing initially
LEARNING_RATE = 3e-4
BATCH_SIZE = 64
N_STEPS = 2048  # Might need adjustment for single env, but keep for now

MODEL_SAVE_DIR = "models"
MODEL_SAVE_NAME = "ppo_negotiator"
LOG_DIR = "logs/ppo_negotiator_logs"

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Custom wrapper function
def create_env_lambda():
    """Create a PettingZoo environment wrapped to be compatible with SB3."""
    # Define agent utilities
    agent_utilities = {
        "agent_0": lambda offer: offer[0] * 0.5 + offer[1] * 0.5,
        "agent_1": lambda offer: offer[0] * 0.5 + offer[1] * 0.5,
    }
    
    # Create the PettingZoo environment
    env_config = {
        "max_rounds": 20,
        "render_mode": None,
        "agent_utils": agent_utilities
    }
    aec_env = negotiation_env(**env_config)
    
    # Wrap it to be compatible with Gym/SB3
    gym_env = PettingZooGymWrapper(aec_env, agent_id="agent_0")
    
    return gym_env

if __name__ == "__main__":
    print("--- Starting RL Agent Training (Single Environment) ---")

    # Create a vectorized environment (even with just one env)
    # DummyVecEnv is used for a single environment
    print("Creating and wrapping environment...")
    
    # Create the wrapped environment
    vec_env = DummyVecEnv([create_env_lambda])
    
    # Wrap with monitoring
    vec_env = VecMonitor(vec_env)
    print("Environment created and wrapped.")

    # --- Model Definition ---
    print(f"Initializing {TRAIN_ALGORITHM.__name__} model with {POLICY_TYPE}...")
    model = TRAIN_ALGORITHM(
        POLICY_TYPE,
        vec_env,
        verbose=1,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        n_steps=N_STEPS,
        tensorboard_log=LOG_DIR,
    )
    print("Model initialized.")

    # --- Training ---
    print(f"Starting training for {TOTAL_TIMESTEPS} timesteps...")
    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            log_interval=1,
            tb_log_name=MODEL_SAVE_NAME,
            reset_num_timesteps=False
        )
        print("Training finished.")

        # --- Save Model ---
        save_path = os.path.join(MODEL_SAVE_DIR, f"{MODEL_SAVE_NAME}_{TOTAL_TIMESTEPS}.zip")
        model.save(save_path)
        print(f"Model saved to {save_path}")
    except KeyError as ke:
        print(f"KeyError during training: {ke}")
        print(traceback.format_exc())  # Print full traceback
    except Exception as e:
        print(f"Training error: {e}")
        print(traceback.format_exc())  # Print full traceback
    finally:
        # --- Cleanup ---
        vec_env.close()
        print("Environment closed.")
        print("--- Training Script Complete ---")

    print("\nTo monitor training with TensorBoard, run:")
    print(f"tensorboard --logdir {LOG_DIR}")