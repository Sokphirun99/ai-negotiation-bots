import random

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
        last_offer = observation.get("last_offer")
        max_utility = observation.get("max_utility", 1.0)
        round_num = observation.get("round", 0)

        if last_offer:
            my_utility = self.utility_function(last_offer)
            # Estimate opponent's utility (CRUCIAL and HARD - requires assumptions or opponent modeling)
            opponent_utility_estimate = self._estimate_opponent_utility(last_offer, max_utility)

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

    def _generate_offer_items(self, target_utility):
        # Placeholder: Needs actual logic based on item space and utility
        print(f"Warning: CooperativeAgent._generate_offer_items needs implementation.")
        # Try to generate an offer that gives roughly target_utility to self
        # and leaves a reasonable amount for the opponent.
        return {"item1": int(target_utility / 2), "item2": int(target_utility / 2)} # Example

    def observe(self, reward, terminated, info):
        """Observes the outcome."""
        pass # Could potentially adjust strategy based on past outcomes