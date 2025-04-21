import os
import numpy as np
from stable_baselines3 import PPO  # Or your specific algorithm
from gymnasium.spaces import utils  # Import space flattening utility

# Assuming NUM_ITEMS and MAX_ITEM_QUANTITY are accessible
try:
    from environment.negotiation_env import NUM_ITEMS, MAX_ITEM_QUANTITY
except ImportError:
    NUM_ITEMS = 2
    MAX_ITEM_QUANTITY = 10

class RLNegotiatorAgent:
    def __init__(self, name="RLAgent", model_path="models/ppo_negotiator"):
        self.name = name
        self.model = None
        self.utility_function = lambda offer: 0.0  # Default utility
        self.env_observation_space = None  # To store env's original space
        self.env_action_space = None  # To store env's original space

        if not model_path.endswith('.zip'):
            model_path += '.zip'

        if os.path.exists(model_path):
            try:
                self.model = PPO.load(model_path)
                print(f"Successfully loaded RL model for {self.name} from {model_path}")
                # --- Store model's expected spaces ---
                print(f"DEBUG ({self.name}): Model Observation Space: {self.model.observation_space}")
                print(f"DEBUG ({self.name}): Model Action Space: {self.model.action_space}")
            except Exception as e:
                print(f"Error loading RL model from {model_path}: {e}")
                self.model = None
        else:
            print(f"Failed to load RL model: Path does not exist at '{model_path}'")
            self.model = None

    def set_utility_function(self, utility_func):
        self.utility_function = utility_func
        print(f"Warning: Setting utility function directly on {self.name} may not be standard.")

    # --- Add method to receive env spaces ---
    def set_environment_spaces(self, observation_space, action_space):
        """Stores the original environment spaces for reference."""
        self.env_observation_space = observation_space
        self.env_action_space = action_space
        print(f"DEBUG ({self.name}): Received Env Observation Space: {self.env_observation_space}")
        print(f"DEBUG ({self.name}): Received Env Action Space: {self.env_action_space}")
    # --- End of added method ---

    def decide(self, observation):
        if not self.model:
            print(f"Error: {self.name} has no loaded model. Returning default action (accept).")
            return {"type": "accept"}

        # --- Add check for env spaces ---
        if self.env_observation_space is None:
            print(f"Error ({self.name}): Environment observation space not set via set_environment_spaces. Cannot preprocess.")
            return {"type": "accept"}
        # --- End of check ---

        print(f"\nDEBUG ({self.name}): Received Raw Env Observation:\n{observation}")  # DEBUG PRINT
        processed_obs = self._preprocess_observation(observation)
        if processed_obs is None:
            print(f"Error: {self.name} failed to preprocess observation. Returning default action.")
            return {"type": "accept"}
        print(f"DEBUG ({self.name}): Preprocessed Observation for Model:\n{processed_obs}")  # DEBUG PRINT

        action, _states = self.model.predict(processed_obs, deterministic=True)
        print(f"DEBUG ({self.name}): Raw Model Action Output:\n{action}")  # DEBUG PRINT

        formatted_action = self._format_action(action)
        print(f"DEBUG ({self.name}): Formatted Action for Env:\n{formatted_action}")  # DEBUG PRINT

        return formatted_action

    def _preprocess_observation(self, env_observation):
        """Converts environment observation to model-compatible format."""
        if self.env_observation_space is None or self.model is None:
            return None

        try:
            # Create a fixed-size observation array matching model's expected shape
            flat_obs = np.zeros(self.model.observation_space.shape, dtype=np.float32)
            
            # Extract only the keys we care about in fixed order
            # 1-2: last_offer_received (2 values)
            if 'last_offer_received' in env_observation and env_observation['last_offer_received'] is not None:
                offer = env_observation['last_offer_received']
                if len(offer) >= 2:  # Ensure there are at least 2 values
                    flat_obs[0] = float(offer[0])
                    flat_obs[1] = float(offer[1])
            
            # 3: offer_valid (1 value)
            if 'offer_valid' in env_observation:
                flat_obs[2] = float(env_observation['offer_valid'])
            
            # 4: round (1 value)
            if 'round' in env_observation:
                round_val = env_observation['round']
                if hasattr(round_val, "__len__"):
                    flat_obs[3] = float(round_val[0])
                else:
                    flat_obs[3] = float(round_val)
            
            # Skip max_utility and last_offer - they weren't part of training
                    
            return flat_obs
            
        except Exception as e:
            print(f"Error during observation preprocessing: {e}")
            return None

    def _format_action(self, model_action):
        """Converts model output to environment-compatible action format."""
        try:
            # Extract action type (first value)
            action_type = int(np.round(model_action[0]))
            action_type = np.clip(action_type, 0, 1)  # Ensure it's 0 or 1

            if action_type == 0:  # Accept
                return {"type": 0}  # Just send type=0 for accept
            else:  # Offer
                # Extract offer values and clip to valid range
                offer_values = np.round(model_action[1:1+NUM_ITEMS]).astype(np.int32)
                offer_values = np.clip(offer_values, 0, MAX_ITEM_QUANTITY)
                
                # Return the correct format with 'offer' key (not 'value')
                return {"type": 1, "offer": offer_values}
        
        except Exception as e:
            print(f"Error formatting action: {e}")
            return {"type": 0}  # Default to accept on error

    def observe(self, reward, terminated, info):
        pass