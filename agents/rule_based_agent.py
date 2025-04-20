import random 

class RuleBasedAgent:
    def __init__(self, name = "RuleBasedAgent", acceptance_threshold = 0.7, offer_range = (0.5, 0.9)):
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
        self.utility_function = lambda itme: sum(itme.values()) # Example utility function

    def set_utility_function(self, utility_func):
        """ Sets the agebt's utility function. """
        self.utility_function = utility_func
    def decide(self, obeservable):
        """
        Decides an action based on the current observation (e.g., opponent's last offer).
        Args:
            observation (dict): Current state, potentially including opponent's offer.
        Returns:
            dict: Action (e.g., {'type': 'offer', 'value': {...}} or {'type': 'accept'})
        """
        last_offer = obeservable.get("last_offer") 
        max_utility = obeservable.get("max_utility", 1.0)

        if last_offer:
            offer_utility = self.utility_function(last_offer) / max_utility
            if offer_utility >= self.acceptance_threshold:
                return {"type": "accept"}
            
        # Make a counter-offer based on its range
        my_offer_utility_ratio = random.uniform(self.offer_range[0], self.offer_range[1])
        # Note: Translating utility ratio back to specific item requires knowledge
        # of the item space and utility function structure, This is simplified.
        # You'd need logic here to construct an actual item bundie offer.
        my_offer_items = self._generate_offer_items(my_offer_utility_ratio * max_utility)
        return {"type": "offer", "value": my_offer_items}
        
        def _generate_offer_items(self, target_utility):
            # Placeholder: Needa actual logic based on item space and utility 
            print(f"Warning: RuleBasedAgent._generate_offer_items needs implementation based on items space.")
            return{"item1":int(target_utility / 2), "item2": int(target_utility / 2)} 
        def observe(self, reward, terminated, info):
            """ Observes the outcome of the last sction (optional for rule-based). """
        pass # Rule-based agents typically don't learn from experience