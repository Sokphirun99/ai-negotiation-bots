import os
import random
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
        self.is_random_agent = False  # Use this as a fallback when model can't be loaded

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
                print(f"WARNING: Falling back to random behavior")
                self.model = None
                self.is_random_agent = True  # If model fails, we'll use random behavior
        else:
            print(f"Failed to load RL model: Path does not exist at '{model_path}'")
            print(f"WARNING: Falling back to random behavior")
            self.model = None
            self.is_random_agent = True  # Use random behavior as fallback

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
        # If model couldn't be loaded and we're using random behavior
        if self.is_random_agent:
            return self._random_action(observation)
            
        if not self.model:
            print(f"Error: {self.name} has no loaded model. Using fallback strategy.")
            return self._random_action(observation)

        # --- Add check for env spaces ---
        if self.env_observation_space is None:
            print(f"Error ({self.name}): Environment observation space not set. Using fallback strategy.")
            return self._random_action(observation)
        # --- End of check ---

        print(f"\nDEBUG ({self.name}): Received Raw Env Observation:\n{observation}")  # DEBUG PRINT
        processed_obs = self._preprocess_observation(observation)
        if processed_obs is None:
            print(f"Error: {self.name} failed to preprocess observation. Using fallback strategy.")
            return self._random_action(observation)
        print(f"DEBUG ({self.name}): Preprocessed Observation for Model:\n{processed_obs}")  # DEBUG PRINT

        try:
            action, _states = self.model.predict(processed_obs, deterministic=True)
            print(f"DEBUG ({self.name}): Raw Model Action Output:\n{action}")  # DEBUG PRINT

            formatted_action = self._format_action(action)
            print(f"DEBUG ({self.name}): Formatted Action for Env:\n{formatted_action}")  # DEBUG PRINT

            return formatted_action
        except Exception as e:
            print(f"Error during model prediction: {e}. Using fallback strategy.")
            return self._random_action(observation)

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
                return {"type": "accept"}  # Format for accept should be "accept" string, not 0
            else:  # Offer
                # Extract offer values and clip to valid range
                offer_values = np.round(model_action[1:1+NUM_ITEMS]).astype(np.int32)
                offer_values = np.clip(offer_values, 0, MAX_ITEM_QUANTITY)
                
                # Format as expected by agent_vs_agent.py - with "value" containing the item allocations
                # This represents the items the agent wants to keep for itself
                my_items = {}
                for i in range(NUM_ITEMS):
                    my_items[f"item{i}"] = int(MAX_ITEM_QUANTITY - offer_values[i])
                
                return {"type": "offer", "value": my_items}
        
        except Exception as e:
            print(f"Error formatting action: {e}")
            return {"type": 0}  # Default to accept on error

    def _random_action(self, observation):
        """Generate random actions when model is not available."""
        print(f"Using random fallback behavior for {self.name}")
        
        # Random decision: 20% chance to accept, 80% chance to make offer
        if 'offer_valid' in observation and observation['offer_valid'] and random.random() < 0.2:
            return {"type": "accept"}
        
        # Make a random offer that keeps most valuable items for self
        my_items = {}
        for i in range(NUM_ITEMS):
            # Keep 60-100% of items for self (give 0-40% to opponent)
            my_share = int(MAX_ITEM_QUANTITY * (0.6 + 0.4 * random.random()))
            my_items[f"item{i}"] = my_share
        
        return {"type": "offer", "value": my_items}

    def observe(self, reward, terminated, info):
        pass