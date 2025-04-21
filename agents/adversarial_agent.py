# filepath: /Users/phirun/Projects/ai-negotiation-bots/agents/adversarial_agent.py
import random
import numpy as np

try:
    from environment.negotiation_env import NUM_ITEMS, MAX_ITEM_QUANTITY
except ImportError:
    print("Warning: Could not import NUM_ITEMS/MAX_ITEM_QUANTITY from environment. Using defaults.")
    NUM_ITEMS = 2
    MAX_ITEM_QUANTITY = 10

class AdversarialAgent:
    def __init__(self, name="AdversarialAgent", high_threshold=0.9, low_offer=0.85):
        """
        Initializes an adversarial agent focused on maximizing own gain.
        Args:
            name (str): Name of the agent.
            high_threshold (float): Minimum utility ratio to accept (very high).
            low_offer (float): Minimum utility ratio to offer (starts high).
        """
        self.name = name
        self.acceptance_threshold = high_threshold
        self.current_offer_ratio = low_offer
        self.utility_function = lambda items: sum(items.values()) # Example utility
        self.concession_step = 0.02 # How much to concede each round if needed
        self.min_offer_ratio = 0.6 # Floor for offers

    def set_utility_function(self, utility_func):
        """Sets the agent's utility function."""
        self.utility_function = utility_func

    def decide(self, observation):
        """
        Decides aggressively to maximize own utility.
        Args:
            observation (dict): Current state, including opponent's offer.
        Returns:
            dict: Action.
        """
        # --- FIX: Update observation key access to match environment format ---
        last_offer_received = observation.get("last_offer_received")
        offer_valid = observation.get("offer_valid", 0)
        
        # Handle round value that could be either an int or a numpy array
        round_val = observation.get("round", 0)
        # If it's a list or numpy array, extract the first element
        if hasattr(round_val, "__len__") and len(round_val) > 0:
            round_value = round_val[0]
        else:
            # Otherwise assume it's already a scalar
            round_value = round_val
        
        # Calculate max possible utility (all items)
        max_items = {f"item{i}": MAX_ITEM_QUANTITY for i in range(NUM_ITEMS)}
        max_utility = self.utility_function(max_items)
        
        # Only process the last offer if it's valid
        if offer_valid and last_offer_received is not None:
            # Convert numpy array to dictionary format for utility function
            last_offer_dict = {f"item{i}": int(last_offer_received[i]) 
                              for i in range(len(last_offer_received))}
            
            offer_utility = self.utility_function(last_offer_dict) / max_utility
            # Accept only if the offer is very high
            if offer_utility >= self.acceptance_threshold:
                return {"type": 0}  # 0 = accept

        # Make a high offer, conceding slightly over time
        offer_ratio = max(self.min_offer_ratio, 
                          self.current_offer_ratio - (round_value * self.concession_step))

        my_offer_items = self._generate_offer_items(offer_ratio * max_utility)
        
        # Debug print to understand agent behavior
        # print(f"DEBUG ({self.name}): Round {round_value}, offering with target ratio {offer_ratio:.2f}")
        # print(f"DEBUG ({self.name}): Offer: {my_offer_items}")
        
        return {"type": 1, "value": my_offer_items}  # 1 = offer

    def _generate_offer_items(self, target_utility_value):
        """
        Generates an item allocation that the agent wants to keep,
        aiming for a specific target utility value.
        (Slightly improved placeholder - still needs proper implementation)
        """
        if not hasattr(self, 'utility_function'):
            print("Error: Utility function not set for agent.")
            return None

        # --- Start of new logic ---
        my_items = np.zeros(NUM_ITEMS, dtype=np.int32)
        current_utility = self.utility_function(my_items)

        # Greedily add items based on potential utility increase (assuming linear utility for simplicity)
        # Create a list of (item_index, potential_utility_gain_per_unit)
        # This requires knowing the utility function structure or estimating gradients.
        # Simple approximation: assume adding one unit of item i gives utility_function([0,..,1,..0])
        item_gains = []
        for i in range(NUM_ITEMS):
             single_item_utility = self.utility_function(np.array([1 if j == i else 0 for j in range(NUM_ITEMS)], dtype=np.int32))
             # Avoid division by zero if single item has 0 utility
             item_gains.append((i, single_item_utility if single_item_utility > 1e-6 else 1e-6))

        # Sort items by potential gain (descending)
        item_gains.sort(key=lambda x: x[1], reverse=True)

        # Iterate through items, adding them until target utility is reached or exceeded
        for item_index, _ in item_gains:
            while my_items[item_index] < MAX_ITEM_QUANTITY:
                if current_utility >= target_utility_value:
                    break # Stop adding this item if target is met/exceeded

                # Try adding one unit
                my_items[item_index] += 1
                new_utility = self.utility_function(my_items)

                # If adding the item decreased utility or didn't change it significantly, undo and stop for this item
                # (This helps with non-linear functions but is basic)
                if new_utility <= current_utility + 1e-6 :
                     my_items[item_index] -= 1 # Undo add
                     break # Stop adding this item

                current_utility = new_utility
            if current_utility >= target_utility_value:
                 break # Stop adding items altogether

        # Convert final numpy array to the dictionary format
        final_offer_dict = {f"item{i}": int(my_items[i]) for i in range(NUM_ITEMS)}
        # print(f"Debug ({self.name}): Generated offer {final_offer_dict} with utility {current_utility:.3f} aiming for {target_utility_value:.3f}")
        return final_offer_dict
        # --- End of new logic ---

    def observe(self, reward, terminated, info):
        """ Observes the outcome (optional). """
        pass