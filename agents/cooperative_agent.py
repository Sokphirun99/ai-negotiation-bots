import random

# Assuming NUM_ITEMS and MAX_ITEM_QUANTITY are accessible or defined here
try:
    # Try importing from environment if it's structured that way
    from environment.negotiation_env import NUM_ITEMS, MAX_ITEM_QUANTITY
except ImportError:
    # Fallback defaults if import fails (adjust if necessary)
    print("Warning: Could not import NUM_ITEMS/MAX_ITEM_QUANTITY from environment. Using defaults.")
    NUM_ITEMS = 2
    MAX_ITEM_QUANTITY = 10

class CooperativeAgent:
    def __init__(self, name="CooperativeAgent", fairness_threshold=0.45, initial_offer_ratio=0.6):
        """
        Initializes a cooperative agent aiming for fair deals.
        Args:
            name (str): Name of the agent.
            fairness_threshold (float): Minimum utility ratio opponent must get for acceptance.
                                       (Requires estimating opponent utility).
            initial_offer_ratio (float): Initial utility ratio the agent offers.
        """
        self.name = name
        self.fairness_threshold = fairness_threshold # How much utility opponent should get
        self.offer_ratio = initial_offer_ratio # Start by offering this utility ratio
        self.utility_function = lambda items: sum(items.values()) # Example utility
        self.patience = 5 # Example: number of rounds before conceding more

    def set_utility_function(self, utility_func):
        """Sets the agent's utility function."""
        self.utility_function = utility_func

    def decide(self, observation):
        """
        Decides based on fairness and cooperation.
        Args:
            observation (dict): Current state, including opponent's offer and round number.
        Returns:
            dict: Action.
        """
        last_offer_received = observation.get("last_offer_received")
        offer_valid = observation.get("offer_valid", 0)
        max_utility = observation.get("max_utility", 1.0)
        
        # Handle round value that could be either an int or a numpy array
        round_val = observation.get("round", 0)
        # If it's a list or numpy array, extract the first element
        if hasattr(round_val, "__len__") and len(round_val) > 0:
            round_num = round_val[0]
        else:
            round_num = round_val

        # Only process if a valid offer was received
        if offer_valid and last_offer_received is not None:
            # Convert numpy array to dictionary format for utility function
            last_offer_dict = {f"item{i}": int(last_offer_received[i]) 
                              for i in range(len(last_offer_received))}
            
            my_utility = self.utility_function(last_offer_dict)
            # Estimate opponent's utility (CRUCIAL and HARD - requires assumptions or opponent modeling)
            opponent_utility_estimate = self._estimate_opponent_utility(last_offer_dict, max_utility)

            my_utility_ratio = my_utility / max_utility
            opponent_utility_ratio_estimate = opponent_utility_estimate / max_utility

            # Accept if the offer seems reasonably fair to both
            if my_utility_ratio >= self.fairness_threshold and opponent_utility_ratio_estimate >= self.fairness_threshold:
                return {"type": "accept"}

        # Concede slightly over time if no agreement
        current_offer_ratio = max(0.5, self.offer_ratio - (round_num / (self.patience * 5))) # Gradually lower demands towards 50%

        my_offer_items = self._generate_offer_items(current_offer_ratio * max_utility)
        return {"type": "offer", "value": my_offer_items}

    def _estimate_opponent_utility(self, offer, max_utility):
        # Placeholder: This is a major challenge. Simplistic assumption:
        # Assume opponent has roughly inverse preferences or similar total utility.
        # A real implementation needs opponent modeling.
        my_utility = self.utility_function(offer)
        # Very naive estimate: assume total utility is constant (max_utility)
        estimated_opponent_utility = max(0, max_utility - my_utility)
        print(f"Warning: CooperativeAgent._estimate_opponent_utility is using a naive estimation.")
        return estimated_opponent_utility

    def _generate_offer_items(self, target_utility_value):
        """
        Generate an offer aiming for a specific utility value.
        This is a more sophisticated implementation that tries to find a fair distribution.
        """
        # Start with an even distribution
        my_items = {}
        for i in range(NUM_ITEMS):
            my_items[f"item{i}"] = MAX_ITEM_QUANTITY // 2
        
        # Adjust until we're close to target utility
        current_utility = self.utility_function(my_items)
        max_iterations = 10
        
        if current_utility > target_utility_value:
            # Need to reduce
            while current_utility > target_utility_value and max_iterations > 0:
                # Reduce my share of a random item
                item_idx = random.randint(0, NUM_ITEMS-1)
                if my_items[f"item{item_idx}"] > 0:
                    my_items[f"item{item_idx}"] -= 1
                    current_utility = self.utility_function(my_items)
                max_iterations -= 1
        else:
            # Need to increase
            while current_utility < target_utility_value and max_iterations > 0:
                # Increase my share of a random item
                item_idx = random.randint(0, NUM_ITEMS-1)
                if my_items[f"item{item_idx}"] < MAX_ITEM_QUANTITY:
                    my_items[f"item{item_idx}"] += 1
                    current_utility = self.utility_function(my_items)
                max_iterations -= 1
        
        # Make sure all values are integers
        for key in my_items:
            my_items[key] = int(my_items[key])
            
        return my_items

    def observe(self, reward, terminated, info):
        """Observes the outcome."""
        pass # Could potentially adjust strategy based on past outcomes