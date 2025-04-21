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
        """
        Converts the environment's observation dictionary into the format
        expected by the trained RL model's observation space.
        (Refined Placeholder)
        """
        if self.env_observation_space is None or self.model is None:
            return None

        # --- IMPORTANT: Use Gymnasium's flatten utility ---
        # This is the standard way SB3's make_vec_env handles Dict spaces for MlpPolicy
        try:
            # Ensure the observation matches the structure defined in the space
            # (This might involve converting lists to numpy arrays if needed)
            # Example: Ensure 'round' is np.array([value]) if space expects Box(1,)
            checked_observation = env_observation.copy()  # Don't modify original
            for key, space in self.env_observation_space.items():
                if key in checked_observation and isinstance(checked_observation[key], list):
                    checked_observation[key] = np.array(checked_observation[key], dtype=space.dtype)
                # Add more checks/conversions if necessary based on your specific Dict structure

            # Flatten the dictionary observation using the environment's space definition
            flat_obs = utils.flatten(self.env_observation_space, checked_observation)
            flat_obs = flat_obs.astype(self.model.observation_space.dtype)  # Ensure correct dtype

            # Double-check shape (optional but good practice)
            if flat_obs.shape != self.model.observation_space.shape:
                print(f"FATAL ERROR ({self.name}): Flattened observation shape {flat_obs.shape} != model expected shape {self.model.observation_space.shape}. Check env space definition and flattening.")
                return None  # Critical mismatch

            return flat_obs

        except Exception as e:
            print(f"Error during observation flattening for {self.name}: {e}")
            print(f"Original Observation: {env_observation}")
            print(f"Env Observation Space: {self.env_observation_space}")
            return None

    def _format_action(self, model_action):
        """
        Converts the action output from the RL model into the dictionary format
        expected by the environment's step function ({'type': int, 'value': dict/None}).
        (Refined Placeholder)
        """
        # This depends HEAVILY on the structure of your environment's action space
        # and how the model's action space corresponds to it.

        # Assuming env action space is Dict:
        # {'type': Discrete(2), 'offer': Box(0, MAX_ITEM_QUANTITY, (NUM_ITEMS,), int32)}
        # And model action space is likely a flattened Box or MultiDiscrete combining these.

        # Example Scenario: Model outputs a single array [type, item0_qty, item1_qty, ...]
        if isinstance(model_action, np.ndarray):
            try:
                # --- Extract action components based on model's action space structure ---
                # This requires knowing the exact structure of self.model.action_space
                # Example: If model.action_space is Box(low=0, high=MAX_Q, shape=(1+NUM_ITEMS,))
                action_type = int(np.round(model_action[0]))  # Round/clip if Box space
                action_type = np.clip(action_type, 0, 1)  # Assuming Discrete(2) for type

                if action_type == 0:  # Accept
                    return {"type": 0}
                elif action_type == 1:  # Offer
                    # Extract offer part - ADJUST indices based on model action space
                    my_items_array = np.round(model_action[1:]).astype(np.int32)
                    my_items_array = np.clip(my_items_array, 0, MAX_ITEM_QUANTITY)

                    # Convert to the required dictionary format for 'value'
                    offer_value = {f"item{i}": int(my_items_array[i]) for i in range(NUM_ITEMS)}
                    return {"type": 1, "value": offer_value}
                else:
                    print(f"Warning ({self.name}): Model output unknown action type {action_type}. Defaulting to accept.")
                    return {"type": 0}

            except (IndexError, ValueError) as e:
                print(f"Error formatting model action array for {self.name}: {e}")
                print(f"Model action received: {model_action}")
                print(f"Model action space: {self.model.action_space if self.model else 'None'}")
                return {"type": 0}  # Default to accept on error
        else:
            print(f"Warning ({self.name}): Unexpected model action format (expected ndarray): {model_action}. Defaulting to accept.")
            return {"type": 0}

    def observe(self, reward, terminated, info):
        pass