import random
import numpy as np

# Assuming NUM_ITEMS and MAX_ITEM_QUANTITY are accessible or defined here
# If not, they might need to be passed during init or accessed differently
try:
    # Try importing from environment if it's structured that way
    from environment.negotiation_env import NUM_ITEMS, MAX_ITEM_QUANTITY
except ImportError:
    # Fallback defaults if import fails (adjust if necessary)
    print("Warning: Could not import NUM_ITEMS/MAX_ITEM_QUANTITY from environment. Using defaults.")
    NUM_ITEMS = 2
    MAX_ITEM_QUANTITY = 10


class RuleBasedAgent:
    def __init__(self, name="RuleBasedAgent", acceptance_threshold=0.7, offer_range=(0.5, 0.9)):
        """
        Initializes a rule-based agent.
        Args:
            name (str): Name of the agent.
            acceptance_threshold (float): Minimum utility ratio to accept an offer.
            offer_range (tuple): Min and max utility ratio for making offers.
        """
        self.name = name
        self.acceptance_threshold = acceptance_threshold
        self.offer_range = offer_range
        self.utility_function = lambda item: sum(item.values())  # Example utility function

    def set_utility_function(self, utility_func):
        """ Sets the agent's utility function. """
        self.utility_function = utility_func

    def decide(self, observation):
        """Makes a decision based on simple rules."""
        my_utility = self.utility_function
        max_utility = observation.get('max_utility', 1.0)  # Get max utility from observation info
        if max_utility == 0:
            max_utility = 1.0  # Avoid division by zero

        last_offer = observation.get('last_offer')  # Offer received from opponent

        if last_offer is not None:
            # Calculate utility of the opponent's last offer
            offer_utility = my_utility(last_offer)
            normalized_offer_utility = offer_utility / max_utility

            # Rule 1: Accept if offer utility exceeds threshold
            if normalized_offer_utility >= self.acceptance_threshold:
                return {"type": "accept"}

        # Rule 2: Make a counter-offer
        # Generate an offer within the agent's desired utility range
        target_utility_ratio = random.uniform(self.offer_range[0], self.offer_range[1])

        # Use the new helper method to generate items based on target utility
        target_utility_value = target_utility_ratio * max_utility
        my_offer_items = self._generate_offer_items(target_utility_value)

        if my_offer_items is None:
            # Fallback if generation fails (e.g., utility too high/low)
            print(f"Warning: {self.name} failed to generate offer for utility {target_utility_value}. Making default offer.")
            # Offer roughly half of everything as a fallback
            default_items = {f"item{i}": MAX_ITEM_QUANTITY // 2 for i in range(NUM_ITEMS)}
            return {"type": "offer", "value": default_items}

        return {"type": "offer", "value": my_offer_items}

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
        """ Observes the outcome of the last action (optional for rule-based). """
        pass  # Rule-based agents typically don't learn from experience