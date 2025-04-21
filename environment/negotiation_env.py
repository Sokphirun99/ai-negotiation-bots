import gymnasium
from gymnasium.spaces import Dict, Box, Discrete
import numpy as np
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers
from pettingzoo.utils.env import ObsType, ActionType

# --- Constants ---
DEFAULT_MAX_ROUNDS = 20
NUM_ITEMS = 2 # Example: negotiating over 2 types of items
MAX_ITEM_QUANTITY = 10 # Example: max 10 units for each item type

def env(**kwargs):
    """
    The env function wraps the environment in wrappers by default.
    """
    internal_env = NegotiationEnv(**kwargs)
    # Add necessary wrappers here if needed (e.g., order enforcing)
    internal_env = wrappers.OrderEnforcingWrapper(internal_env)
    return internal_env

class NegotiationEnv(AECEnv):
    """
    PettingZoo environment for a two-agent negotiation over a set of items.

    Follows the AEC API (Agent Environment Cycle).
    Agents take turns making offers or accepting.
    """
    metadata = {
        "name": "negotiation_v0",
        "render_modes": ["human"],
        "is_parallelizable": False,
    }

    def __init__(
        self,
        max_rounds: int = DEFAULT_MAX_ROUNDS,
        render_mode: str | None = None,
        agent_utils: dict | None = None, # Dict mapping agent_id -> utility_function
    ):
        super().__init__()
        self.max_rounds = max_rounds
        self.render_mode = render_mode

        # --- Agent Setup ---
        self.possible_agents = ["agent_0", "agent_1"]
        self.agent_name_mapping = dict(zip(self.possible_agents, list(range(len(self.possible_agents)))))
        self._agent_selector = agent_selector(self.possible_agents)

        # --- Utility Functions ---
        # Example: Simple linear utility functions if not provided
        # agent_utils = {"agent_0": lambda x: x[0]*0.7 + x[1]*0.3, "agent_1": lambda x: x[0]*0.3 + x[1]*0.7}
        self.utility_functions = agent_utils if agent_utils else self._default_utility_funcs()
        self.max_utilities = self._calculate_max_utilities()


        # --- State Variables ---
        self.round_num = 0
        self.last_offers = {agent: None for agent in self.possible_agents} # Stores the last offer *made by* the agent
        self.current_proposer = None # Who is making the offer this turn
        self.current_responder = None # Who is responding to the offer

        # --- Spaces ---
        # Observation space: round, last offer received (if any)
        # Represent offer as Box: [item1_qty_for_me, item2_qty_for_me]
        # Add round number
        self._obs_space = Dict({
            "round": Box(low=0, high=self.max_rounds, shape=(1,), dtype=np.int32),
            "last_offer_received": Box(low=0, high=MAX_ITEM_QUANTITY, shape=(NUM_ITEMS,), dtype=np.int32),
            "offer_valid": Discrete(2) # 0 if no valid offer received yet, 1 otherwise
        })

        # Action space:
        # 0: Accept last offer
        # 1: Make counter-offer (represented by the Dict space)
        # Offer Dict: {item_i: quantity_for_opponent}
        # Note: Agent decides how many items to give to the *opponent*.
        # The agent implicitly keeps the rest.
        self._action_space = Dict({
            "type": Discrete(2), # 0=Accept, 1=Offer
            "offer": Box(low=0, high=MAX_ITEM_QUANTITY, shape=(NUM_ITEMS,), dtype=np.int32)
        })

        # PettingZoo API requirements
        self.agents = []
        self.rewards = {agent: 0 for agent in self.possible_agents}
        self._cumulative_rewards = {agent: 0 for agent in self.possible_agents}
        self.terminations = {agent: False for agent in self.possible_agents}
        self.truncations = {agent: False for agent in self.possible_agents}
        self.infos = {agent: {} for agent in self.possible_agents}

    def _default_utility_funcs(self):
        # Simple linear functions, agent 0 values item 0 more, agent 1 values item 1 more
        print("Warning: Using default utility functions.")
        utils = {
            "agent_0": lambda offer: offer[0] * 0.7 + offer[1] * 0.3,
            "agent_1": lambda offer: offer[0] * 0.3 + offer[1] * 0.7,
        }
        return utils

    def _calculate_max_utilities(self):
        # Calculate max possible utility for each agent (if they got all items)
        all_items = np.array([MAX_ITEM_QUANTITY] * NUM_ITEMS, dtype=np.int32)
        max_utils = {
            agent: self.utility_functions[agent](all_items)
            for agent in self.possible_agents
        }
        # Avoid division by zero if max utility is 0
        for agent in max_utils:
            if max_utils[agent] == 0:
                max_utils[agent] = 1.0
        return max_utils

    def observation_space(self, agent: str) -> gymnasium.spaces.Space:
        return self._obs_space

    def action_space(self, agent: str) -> gymnasium.spaces.Space:
        return self._action_space

    def observe(self, agent: str) -> ObsType:
        """Generates the observation for the specified agent."""
        opponent = self.possible_agents[1 - self.agent_name_mapping[agent]]
        last_offer_made_by_opponent = self.last_offers[opponent]

        obs = {
            "round": np.array([self.round_num], dtype=np.int32),
            "last_offer_received": np.zeros(NUM_ITEMS, dtype=np.int32), # Default if no offer
            "offer_valid": 0
        }

        if last_offer_made_by_opponent is not None:
            # The offer stored is what the opponent proposed *for the current agent*
            obs["last_offer_received"] = last_offer_made_by_opponent
            obs["offer_valid"] = 1

        return obs

    def _get_info(self, agent: str) -> dict:
        """Generates info dictionary for the agent."""
        return {
            "round": self.round_num,
            "last_offer_made": self.last_offers[agent],
            "opponent_last_offer": self.last_offers[self.possible_agents[1 - self.agent_name_mapping[agent]]],
            "max_utility": self.max_utilities[agent]
        }

    def reset(self, seed: int | None = None, options: dict | None = None) -> None:
        """Resets the environment to a starting state."""
        if seed is not None:
            np.random.seed(seed) # Seed the random number generator if needed

        self.agents = self.possible_agents[:]
        self._agent_selector.reinit(self.agents)
        self.agent_selection = self._agent_selector.next() # Selects the first agent

        self.round_num = 0
        self.last_offers = {agent: None for agent in self.possible_agents}
        self.current_proposer = self.agent_selection
        self.current_responder = None # No responder initially

        # Reset API required variables
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: self._get_info(agent) for agent in self.agents}


    def _is_valid_offer(self, offer_to_opponent: np.ndarray) -> bool:
        """Checks if an offer (items given to opponent) is valid."""
        # Check non-negative
        if np.any(offer_to_opponent < 0):
            return False
        # Check if total items allocated exceed available quantity
        items_for_proposer = np.array([MAX_ITEM_QUANTITY] * NUM_ITEMS) - offer_to_opponent
        if np.any(items_for_proposer < 0): # Opponent asking for more than exists
             return False
        # Check if proposer keeps non-negative items
        # (This is implicitly covered by the previous check)

        return True

    def step(self, action: ActionType) -> None:
        """Processes the action of the current agent."""
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            # Handle dead agent stepping (required by API)
            self._was_dead_step(action)
            return

        agent = self.agent_selection
        self._cumulative_rewards[agent] = 0 # Reset reward for current step

        # --- Process Action ---
        action_type = action["type"]
        offer_to_opponent = action["offer"] # What the current agent offers *to the opponent*

        agreement_reached = False
        final_distribution = None # Stores the agreed distribution {agent: items_array}

        # Action 0: Accept
        if action_type == 0:
            opponent = self.possible_agents[1 - self.agent_name_mapping[agent]]
            last_offer_by_opponent = self.last_offers[opponent]
            if last_offer_by_opponent is not None:
                # Agreement reached! The last offer made by the opponent is accepted.
                # The offer was specified as items *for the current agent*.
                agreement_reached = True
                items_for_current_agent = last_offer_by_opponent
                items_for_opponent = np.array([MAX_ITEM_QUANTITY] * NUM_ITEMS) - items_for_current_agent
                final_distribution = {
                    agent: items_for_current_agent,
                    opponent: items_for_opponent
                }
                print(f"Round {self.round_num}: {agent} ACCEPTS offer from {opponent}. Distribution: {final_distribution}")

            else:
                # Invalid action: Cannot accept if no offer was made
                print(f"Round {self.round_num}: {agent} tried to ACCEPT, but no previous offer exists. Treating as invalid.")
                # Penalize? Or just treat as making a minimal offer? For now, just end turn.
                pass # Agent loses turn essentially

        # Action 1: Make Offer
        elif action_type == 1:
            if self._is_valid_offer(offer_to_opponent):
                # Store the offer in terms of what the *recipient* (next agent) will get.
                items_for_recipient = np.array([MAX_ITEM_QUANTITY] * NUM_ITEMS) - offer_to_opponent
                self.last_offers[agent] = items_for_recipient # Store what agent offers *to the opponent*
                print(f"Round {self.round_num}: {agent} OFFERS {items_for_recipient} to opponent.")
            else:
                # Invalid offer made
                print(f"Round {self.round_num}: {agent} made INVALID offer: {offer_to_opponent}. Treating as invalid.")
                # Penalize? For now, just end turn. Agent loses turn.
                self.last_offers[agent] = None # Invalidate previous offer if any? Or keep old one? Let's clear.
                pass

        # --- Update State and Rewards ---
        if agreement_reached:
            # Calculate utility based on the agreed distribution
            utility_agent = self.utility_functions[agent](final_distribution[agent])
            utility_opponent = self.utility_functions[opponent](final_distribution[opponent])

            # Reward based on normalized utility (0 to 1)
            self.rewards[agent] = utility_agent / self.max_utilities[agent]
            self.rewards[opponent] = utility_opponent / self.max_utilities[opponent]

            # Terminate for all agents
            self.terminations = {a: True for a in self.agents}
            print(f"Agreement reached! Utilities: {agent}={utility_agent:.2f}, {opponent}={utility_opponent:.2f}")
            print(f"Normalized Rewards: {agent}={self.rewards[agent]:.2f}, {opponent}={self.rewards[opponent]:.2f}")

        else:
            # Small penalty for taking a step without agreement? Optional.
            # self.rewards[agent] = -0.01
            pass


        # --- Advance Round and Check Truncation ---
        # Cycle agent - select next agent
        self.agent_selection = self._agent_selector.next()

        # Check if a full round of offers has passed (both agents acted)
        if self._agent_selector.is_last():
            self.round_num += 1
            print(f"--- End of Round {self.round_num-1} ---")


        # Truncate if max rounds reached
        if self.round_num >= self.max_rounds:
            self.truncations = {a: True for a in self.agents}
            print(f"Max rounds ({self.max_rounds}) reached. Negotiation failed.")
            # Assign low reward for failure?
            self.rewards = {a: -1.0 for a in self.agents} # Example failure penalty


        # Update cumulative rewards and infos
        for ag in self.agents:
            self._cumulative_rewards[ag] += self.rewards[ag]
            self.infos[ag] = self._get_info(ag) # Update info for all agents

        # Render if requested
        if self.render_mode == "human":
            self.render()

    def render(self) -> None:
        """Renders the environment state (optional)."""
        if self.render_mode == "human":
            print("-" * 20)
            print(f"Round: {self.round_num}")
            print(f"Current Agent: {self.agent_selection}")
            for agent_id in self.possible_agents:
                 offer = self.last_offers[agent_id]
                 print(f"  {agent_id} Last Offer Made (to opponent): {offer if offer is not None else 'None'}")
            print("-" * 20)

    def close(self) -> None:
        """Cleans up resources."""
        pass # Nothing specific to close here for now

# Example Usage (can be run directly for basic checks)
if __name__ == "__main__":
    env = NegotiationEnv(render_mode="human")
    env.reset()

    print("Observation Spaces:")
    for agent in env.possible_agents:
        print(f"{agent}: {env.observation_space(agent)}")

    print("\nAction Spaces:")
    for agent in env.possible_agents:
        print(f"{agent}: {env.action_space(agent)}")

    print("\nStarting Simulation...")

    # Manual stepping example (replace with agent logic)
    for agent in env.agent_iter(max_iter=DEFAULT_MAX_ROUNDS * len(env.possible_agents) + 1):
        observation, reward, termination, truncation, info = env.last()

        if termination or truncation:
            print(f"Agent {agent} terminated or truncated.")
            action = None # Agent is done
        else:
            # --- Replace with actual agent decision logic ---
            print(f"\nAgent {agent}'s turn (Round {info.get('round', -1)})")
            print(f"Observation: {observation}")
            # Example: Simple rule-based action (Agent 0 always offers [5,5], Agent 1 accepts if offer >= [3,3])
            if agent == "agent_0":
                action = {"type": 1, "offer": np.array([5, 5], dtype=np.int32)} # Offer 5,5 to agent_1
                print(f"Agent {agent} decides: Offer {action['offer']}")
            else: # agent_1
                last_offer_rcvd = observation["last_offer_received"]
                offer_valid = observation["offer_valid"]
                if offer_valid and np.all(last_offer_rcvd >= np.array([3, 3])):
                     action = {"type": 0, "offer": np.zeros(NUM_ITEMS, dtype=np.int32)} # Accept (offer part ignored)
                     print(f"Agent {agent} decides: Accept")
                else:
                     # Make a counter-offer (e.g., offer [4,6] to agent_0)
                     action = {"type": 1, "offer": np.array([4, 6], dtype=np.int32)}
                     print(f"Agent {agent} decides: Offer {action['offer']}")
            # ------------------------------------------------

        env.step(action)

    env.close()
    print("\nSimulation Finished.")