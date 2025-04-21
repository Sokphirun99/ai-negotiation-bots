from stable_baselines3 import PPO
import numpy as np

class RLNegotiatorAgent:
    def __init__(self, model_path, name="RLNegotiatorAgent"):
        """
        Initializes an RL-based agent by loading a pre-trained model.
        Args:
            model_path (str): Path to the trained Stable-Baselines3 model.
            name (str): Name of the agent.
        """
        self.name = name
        try:
            self.model = PPO.load(model_path)
            print(f"RL model loaded successfully from {model_path}")
        except Exception as e:
            print(f"Failed to load RL model from {model_path}: {e}")
            print("RLNrgotistorAgent will not function correctly.")
            self.action_space = None
                # Utility function might be implicitly learned or handled by the environment's reward
        self.utility_function = None # Usually not explicitly defined here for RL

    def set_utility_function(self, utility_func):
                 # RL agents typically learn based on rewards from the environment,
                 # which reflect the utility. Direct setting might not be standard.
        print("Warning: Setting utility function directly on RL agent may not be standard.")
        self.utility_function = utility_func

    def decide(self, observation):
        """
        Decides an action using the loaded RL model.
        Args:
            observation: The current environment state, formatted as expected by the model.
                         Needs to be compatible with the model's observation space.
        Returns:
            Action predicted by the RL model. The format depends on the action space.
        """
        if not self.model:
            print("Error: RL model not loaded. Cannot decide.")
                        # Return a default/dummy action or raise an error
            return {"type": "no_action"} # Example dummy action

        # Ensure observation is in the correct format (e.g., numpy array)
        # This step is CRUCIAL and depends heavily on your environment definition
        processed_observation = self._preprocess_observation(observation)

        action, _states = self._preprocess_observation(observation)
                # Post-process action if necessary to fit the environment's expected format
        formatted_action = self._format_action(action)
        return formatted_action
    def _preprocess_observation(self, observation):
        # Placeholder: Convert your environment's observation dict/object
        # into the format expected by the SB3 model (e.g., a NumPy array).
        # This depends heavily on your NegotiationEnv's observation space.
        print("Warning: RLNegotiatorAgent._preprocess_observation needs implementation.")
        # Example: Flatten dictionary values into a numpy array
        # This assumes observation is a dict of numbers and order is consistent.
        try:
            # Ensure consistent order if using dict values
            # It's generally better to define an explicit order or use Gymnasium spaces
            ordered_keys = sorted(observation.key())
            return np.array(observation.keys())
        except Exception as e:
            print(f"Error processing observation: {e}. Using zeros.")
            # Determine the expected shape from the model if possible
            obs_shape = self.model.observation_space.shape if self.model else (1,)
            return np.zeros(obs_shape, dtype=np. float32)
        
    def _format_action(self, action):
        # Placeholder: Convert the action output by the SB3 model
        # into the format expected by your environment (e.g., a dictionary).
        # This depends heavily on your NegotiationEnv's action space.
        print("Warning: RLNegotiatorAgent._format_action needs implementation.")
        # Example: If action is a single number representing offer utility ratio
        if isinstance(action, (np.number, float)):
            return {"type": "offer", "value": self._generate_offer_item(action)}
        # Example: If action is discrete (e.g., 0=accept, 1=reject, 2=offer_low, 3=offer_high)
        elif isinstance(action, (np.int64, int)):
            if action == 0: return {"type": "accept"}
            elif action == 1: return {"type": "reject"} # Or make a counter-offer
            else:
                # Needs logic for different offer types
                offer_val = 0.5 if action == 2 else 0.8 # Example
                return {"type": "offer", "value": self._generate_offer_item(offer_val)}
        return action # Return raw action if no formatting needed/possible
    
    def _generate_offer_item(self, target_utility_ratio):
        # Placeholder: Needs actual logic based on item space and utility
        print(f"Warning: RLNegotiatorAgent._generate_offer_items needs implementation.")
        # This likely needs access to max_utility or item details
        return {"item1": int(target_utility_ratio * 5), "item2": int(target_utility_ratio * 5)} 

    def observe(self, reward, terminated, info):
        """Observes the outcome (RL agents use this during training, not typically during inference)."""
        # This method is primarily used during the training loop, not usually
        # when the agent is just acting based on a loaded policy.
        pass