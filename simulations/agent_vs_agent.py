import argparse
import os
import sys
import numpy as np
import pandas as pd

# Add project root to Python path to import modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import environment and agent classes
from environment.negotiation_env import env as negotiation_env, MAX_ITEM_QUANTITY, NUM_ITEMS
from agents.rule_based_agent import RuleBasedAgent
from agents.adversarial_agent import AdversarialAgent
from agents.cooperative_agent import CooperativeAgent
from agents.rl_negotiator_agent import RLNegotiatorAgent

# --- Agent Mapping ---
AGENT_MAP = {
    "rulebased": RuleBasedAgent,
    "adversarial": AdversarialAgent,
    "cooperative": CooperativeAgent,
    "rl": RLNegotiatorAgent,
}

# --- Simulation Function (adapted from main.py) ---
def run_simulation(env, agents_dict):
    """Runs a single negotiation simulation episode."""
    print("\n--- Starting New Simulation ---")
    env.reset()
    env_utils = env.unwrapped.utility_functions
    for agent_id, agent_instance in agents_dict.items():
        if hasattr(agent_instance, 'set_utility_function') and agent_id in env_utils:
            agent_instance.set_utility_function(env_utils[agent_id])

    agreement_reached = False
    final_rewards = {}
    final_infos = {}
    log_entries = []  # To store step-by-step data

    for agent_id in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        step_log = {
            'agent_id': agent_id,
            'round': info.get('round', -1),
            'reward': reward,
            'termination': termination,
            'truncation': truncation,
            'action_type': None,
            'offer_to_opponent': None,
        }

        # Initialize env_action for this step
        env_action = None  # Default to None

        if termination or truncation:
            final_rewards[agent_id] = env.rewards[agent_id]
            final_infos[agent_id] = info
            if termination and reward > 0:
                agreement_reached = True
        else:
            # Agent is alive, calculate action
            current_agent = agents_dict[agent_id]
            agent_observation = observation.copy()  # Use copy to avoid modifying original
            agent_observation['max_utility'] = info.get('max_utility', 1.0)
            agent_observation['round'] = info.get('round', 0)
            if observation['offer_valid']:
                agent_observation['last_offer'] = observation['last_offer_received']
            else:
                agent_observation['last_offer'] = None

            action_dict = current_agent.decide(agent_observation)

            # Format action (only if agent is alive)
            env_action = {"type": 0, "offer": np.zeros(NUM_ITEMS, dtype=np.int32)}
            if isinstance(action_dict, dict):
                # Convert string action type to integer for environment
                if action_dict.get("type") == "accept":
                    env_action["type"] = 0
                elif action_dict.get("type") == "offer":
                    env_action["type"] = 1
                else:
                    # Fall back to numeric values if provided
                    env_action["type"] = action_dict.get("type", 0)
                
                agent_offer_value = action_dict.get("value")
                step_log['action_type'] = env_action["type"]  # Log action type

                if env_action["type"] == 1 and agent_offer_value is not None:
                    my_items_array = np.zeros(NUM_ITEMS, dtype=np.int32)
                    total_items_array = np.array([MAX_ITEM_QUANTITY] * NUM_ITEMS, dtype=np.int32)
                    try:
                        for i in range(NUM_ITEMS):
                            item_key = f"item{i}"
                            my_items_array[i] = agent_offer_value.get(item_key, 0)
                        my_items_array = np.minimum(my_items_array, total_items_array)
                        offer_to_opponent = total_items_array - my_items_array
                        env_action["offer"] = offer_to_opponent
                        step_log['offer_to_opponent'] = offer_to_opponent.tolist()  # Log offer
                    except Exception as e:
                        print(f"Error formatting offer from {agent_id}: {e}. Sending default.")
                        env_action["offer"] = np.array([MAX_ITEM_QUANTITY // 2] * NUM_ITEMS, dtype=np.int32)
                        step_log['offer_to_opponent'] = env_action["offer"].tolist()
            else:
                print(f"Error: Agent {agent_id} returned non-dict action: {action_dict}")
                env_action = {"type": 1, "offer": np.array([1] * NUM_ITEMS, dtype=np.int32)}

        log_entries.append(step_log)
        env.step(env_action)

    print("--- Simulation Finished ---")
    print(f"Agreement Reached: {agreement_reached}")
    print(f"Final Rewards: {final_rewards}")

    env.close()
    return final_rewards, final_infos, agreement_reached, log_entries


# --- Argument Parsing and Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AI Negotiation Simulation")
    parser.add_argument("--agent1", type=str, required=True, choices=AGENT_MAP.keys(),
                        help="Type of the first agent.")
    parser.add_argument("--agent2", type=str, required=True, choices=AGENT_MAP.keys(),
                        help="Type of the second agent.")
    parser.add_argument("--agent1_config", type=str, default="{}",
                        help="JSON string for agent1 constructor arguments.")
    parser.add_argument("--agent2_config", type=str, default="{}",
                        help="JSON string for agent2 constructor arguments.")
    parser.add_argument("--rl_model_path", type=str, default="models/ppo_negotiator_100000.zip",
                        help="Path to the saved RL model zip file (used if agent type is 'rl').")
    parser.add_argument("--rounds", type=int, default=20,
                        help="Maximum number of negotiation rounds.")
    parser.add_argument("--simulations", type=int, default=1,
                        help="Number of simulations to run.")
    parser.add_argument("--log_dir", type=str, default="logs/simulation_logs",
                        help="Directory to save simulation logs.")
    parser.add_argument("--render", action="store_true",
                        help="Render the environment steps (human mode).")

    args = parser.parse_args()

    # --- Environment Setup ---
    render_mode = "human" if args.render else None
    # Define utility functions (example - could be configurable)
    agent_utilities = {
        "agent_0": lambda offer: offer[0] * 0.9 + offer[1] * 0.1 if isinstance(offer, (np.ndarray, list)) 
                    else offer.get("item0", 0) * 0.9 + offer.get("item1", 0) * 0.1,
        
        "agent_1": lambda offer: offer[0] * 0.1 + offer[1] * 0.9 if isinstance(offer, (np.ndarray, list))
                    else offer.get("item0", 0) * 0.1 + offer.get("item1", 0) * 0.9,
    }
    env = negotiation_env(render_mode=render_mode, max_rounds=args.rounds, agent_utils=agent_utilities)

    # --- Initialize Agents ---
    try:
        import json
        agent1_kwargs = json.loads(args.agent1_config)
        agent2_kwargs = json.loads(args.agent2_config)
    except json.JSONDecodeError as e:
        print(f"Error parsing agent config JSON: {e}")
        print("Please provide valid JSON strings for --agent1_config and --agent2_config.")
        sys.exit(1)

    Agent1Class = AGENT_MAP[args.agent1]
    Agent2Class = AGENT_MAP[args.agent2]

    # Special handling for RL agent model path
    if Agent1Class == RLNegotiatorAgent:
        agent1_kwargs['model_path'] = args.rl_model_path
    if Agent2Class == RLNegotiatorAgent:
        agent2_kwargs['model_path'] = args.rl_model_path

    try:
        agent0 = Agent1Class(**agent1_kwargs)
        agent1 = Agent2Class(**agent2_kwargs)

        # --- ADD THIS: Set env spaces for RL agents ---
        if isinstance(agent0, RLNegotiatorAgent):
            agent0.set_environment_spaces(env.observation_space("agent_0"), env.action_space("agent_0"))
        if isinstance(agent1, RLNegotiatorAgent):
            agent1.set_environment_spaces(env.observation_space("agent_1"), env.action_space("agent_1"))
        # --- END OF ADDED CODE ---

        # Debug print for model loading check
        if args.agent1 == 'rl':
            print(f"DEBUG: Agent 0 (RL) model object: {agent0.model}")
        if args.agent2 == 'rl':
            print(f"DEBUG: Agent 1 (RL) model object: {agent1.model}")
    except TypeError as e:
        print(f"Error initializing agents. Check config arguments: {e}")
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"Error loading RL model: {e}")
        print(f"Ensure model exists at: {args.rl_model_path}")
        sys.exit(1)

    agents = {
        "agent_0": agent0,
        "agent_1": agent1,
    }

    # --- Run Simulation(s) ---
    all_results = []
    os.makedirs(args.log_dir, exist_ok=True)

    for i in range(args.simulations):
        print(f"\n--- Running Simulation {i+1}/{args.simulations} ---")
        final_rewards, final_infos, agreement, log_data = run_simulation(env, agents)

        # --- Log Results ---
        result_summary = {
            'simulation_id': i,
            'agent1_type': args.agent1,
            'agent2_type': args.agent2,
            'agreement_reached': agreement,
            'agent0_final_reward': final_rewards.get('agent_0', np.nan),
            'agent1_final_reward': final_rewards.get('agent_1', np.nan),
            'final_round': final_infos.get('agent_0', {}).get('round', -1)
        }
        all_results.append(result_summary)

        # Save detailed step log for this simulation
        log_df = pd.DataFrame(log_data)
        log_filename = os.path.join(args.log_dir, f"sim_{i}_A0_{args.agent1}_vs_A1_{args.agent2}_steps.csv")
        log_df.to_csv(log_filename, index=False)
        print(f"Detailed log saved to: {log_filename}")

    # Save summary results
    summary_df = pd.DataFrame(all_results)
    summary_filename = os.path.join(args.log_dir, f"summary_A0_{args.agent1}_vs_A1_{args.agent2}_{args.simulations}_sims.csv")
    summary_df.to_csv(summary_filename, index=False)
    print(f"\nSummary results saved to: {summary_filename}")

    # Print aggregate summary
    if args.simulations > 1:
        print("\n--- Aggregate Results ---")
        agreement_rate = summary_df['agreement_reached'].mean()
        avg_reward_a0 = summary_df['agent0_final_reward'].mean()
        avg_reward_a1 = summary_df['agent1_final_reward'].mean()
        print(f"Agreement Rate: {agreement_rate:.2%}")
        print(f"Avg Reward Agent 0 ({args.agent1}): {avg_reward_a0:.3f}")
        print(f"Avg Reward Agent 1 ({args.agent2}): {avg_reward_a1:.3f}")

    # Example configuration
    example_config = {"high_threshold": 0.7, "low_offer": 0.7, "concession_step": 0.05}
