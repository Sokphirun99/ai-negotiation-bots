import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import json
import subprocess
from matplotlib.colors import ListedColormap

st.set_page_config(page_title="AI Negotiation Bots", layout="wide")

# --- Header ---
st.title("🤝 AI Negotiation Bot Simulator")
st.markdown("Run simulations between different negotiating agents and visualize the results.")

# --- Sidebar Configuration ---
st.sidebar.header("Simulation Settings")

# Agent selection
agent_types = ["rulebased", "adversarial", "cooperative", "rl"]
agent1 = st.sidebar.selectbox("Agent 1 (agent_0)", agent_types, index=3)  # Default to RL
agent2 = st.sidebar.selectbox("Agent 2 (agent_1)", agent_types, index=1)  # Default to adversarial

# Agent configuration (advanced)
with st.sidebar.expander("Agent 1 Configuration"):
    agent1_config = st.text_area("Agent 1 JSON Config", "{}")
    
with st.sidebar.expander("Agent 2 Configuration"):
    agent2_config = st.text_area("Agent 2 JSON Config", "{}")

# RL model path (if applicable)
if agent1 == "rl" or agent2 == "rl":
    available_models = [f for f in os.listdir("models") if f.endswith(".zip")]
    default_model = "ppo_negotiator_10000.zip" if "ppo_negotiator_10000.zip" in available_models else available_models[0] if available_models else ""
    
    rl_model = st.sidebar.selectbox(
        "RL Model Path", 
        available_models,
        index=available_models.index(default_model) if default_model in available_models else 0
    )
    rl_model_path = os.path.join("models", rl_model)
else:
    rl_model_path = "models/ppo_negotiator_10000.zip"  # Default path

# Utility function type selection
utility_type = st.sidebar.selectbox(
    "Utility Function Type", 
    ["Complementary", "Competitive", "Balanced", "Super-additive"]
)
# Then map this selection to different utility functions

# Simulation parameters
rounds = st.sidebar.slider("Max Rounds", min_value=5, max_value=50, value=20)
num_sims = st.sidebar.slider("Number of Simulations", min_value=1, max_value=20, value=1)

# --- Run Simulation Button ---
run_simulation = st.sidebar.button("Run Simulation", type="primary")

# Main content area
col1, col2 = st.columns([3, 2])

# Display agent descriptions
with col1:
    st.subheader("Agent Strategies")
    agent_descriptions = {
        "rulebased": "Uses fixed thresholds for acceptance and offering, with minimal adaptation.",
        "adversarial": "Prioritizes maximizing its own utility, willing to risk negotiation failure.",
        "cooperative": "Aims for fair deals that satisfy both parties, more willing to compromise.",
        "rl": "Trained using Reinforcement Learning (PPO) to maximize long-term rewards."
    }
    
    st.info(f"**Agent 1:** {agent1.title()} - {agent_descriptions[agent1]}")
    st.warning(f"**Agent 2:** {agent2.title()} - {agent_descriptions[agent2]}")
    
    # Agent preferences visualization
    st.subheader("Agent Utility Preferences")
    preferences = pd.DataFrame({
        "Item": ["Item 0", "Item 1"],
        "Agent 1 Preference": [0.9, 0.1],
        "Agent 2 Preference": [0.1, 0.9]
    })
    
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(preferences["Item"]))
    width = 0.35
    
    ax.bar(x - width/2, preferences["Agent 1 Preference"], width, label="Agent 1", color="blue", alpha=0.7)
    ax.bar(x + width/2, preferences["Agent 2 Preference"], width, label="Agent 2", color="orange", alpha=0.7)
    
    ax.set_ylabel("Preference Weight")
    ax.set_title("Agent Utility Preferences")
    ax.set_xticks(x)
    ax.set_xticklabels(preferences["Item"])
    ax.legend()
    ax.set_ylim(0, 1)
    
    st.pyplot(fig)

# Results area (initially hidden)
results_container = col2.container()

if run_simulation:
    with st.spinner(f"Running {num_sims} simulation(s)..."):
        # Build command
        cmd = [
            "python", "simulations/agent_vs_agent.py",
            "--agent1", agent1,
            "--agent2", agent2,
            "--agent1_config", agent1_config,
            "--agent2_config", agent2_config,
            "--rl_model_path", rl_model_path,
            "--rounds", str(rounds),
            "--simulations", str(num_sims)
        ]
        
        # Run simulation
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        stdout, stderr = process.communicate()
        
        # Debug output
        with st.expander("Command Output"):
            st.code(stdout)
            if stderr:
                st.error(stderr)

    # Load and display results
    with results_container:
        st.subheader("Simulation Results")
        
        # Try to find the most recent summary file
        log_dir = "logs/simulation_logs"
        summary_files = [f for f in os.listdir(log_dir) if f.startswith(f"summary_A0_{agent1}_vs_A1_{agent2}")]
        if summary_files:
            latest_summary = max(summary_files, key=lambda x: os.path.getmtime(os.path.join(log_dir, x)))
            summary_path = os.path.join(log_dir, latest_summary)
            
            # Load summary data
            summary_df = pd.read_csv(summary_path)
            
            # Overall statistics
            agreement_rate = summary_df['agreement_reached'].mean() * 100
            avg_reward_a0 = summary_df['agent0_final_reward'].mean()
            avg_reward_a1 = summary_df['agent1_final_reward'].mean()
            
            # Display metrics
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Agreement Rate", f"{agreement_rate:.1f}%")
            col_b.metric("Avg Reward Agent 1", f"{avg_reward_a0:.2f}")
            col_c.metric("Avg Reward Agent 2", f"{avg_reward_a1:.2f}")
            
            # Load most recent simulation log for details
            step_files = [f for f in os.listdir(log_dir) if f.startswith(f"sim_") and f.endswith("_steps.csv")]
            if step_files:
                latest_step = max(step_files, key=lambda x: os.path.getmtime(os.path.join(log_dir, x)))
                step_path = os.path.join(log_dir, latest_step)
                steps_df = pd.read_csv(step_path)
                
                # Visualize negotiation process
                st.subheader("Negotiation Process Visualization")
                
                # Find the final round
                max_round = int(steps_df['round'].max())
                
                # Create a visualization of the negotiation
                # (This could be improved with more details)
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Plot rounds on x-axis
                rounds_x = np.arange(max_round + 1)
                
                # Create a colorful timeline
                cmap = ListedColormap(['lightblue', 'lightgreen'])
                ax.imshow(np.zeros((1, max_round + 1)), cmap=cmap, aspect='auto', alpha=0.3)
                
                # Add negotiation events
                for _, row in steps_df.iterrows():
                    if pd.notna(row['action_type']):
                        action = "ACCEPT" if row['action_type'] == 0 else "OFFER"
                        agent_idx = 0 if row['agent_id'] == 'agent_0' else 1
                        marker = 'o' if action == "ACCEPT" else '^'
                        color = 'green' if action == "ACCEPT" else 'blue' if agent_idx == 0 else 'orange'
                        
                        ax.scatter(row['round'], agent_idx, marker=marker, s=100, 
                                  color=color, label=f"{row['agent_id']} {action}")
                        
                        # Add offer details for offers
                        if action == "OFFER" and isinstance(row['offer_to_opponent'], str):
                            try:
                                # Handle various formats of offer data
                                if row['offer_to_opponent'].startswith('['):
                                    # List format
                                    offer = eval(row['offer_to_opponent'])
                                    offer_text = f"[{','.join(map(str, offer))}]"
                                else:
                                    offer_text = row['offer_to_opponent']
                                
                                ax.annotate(offer_text, 
                                           (row['round'], agent_idx),
                                           xytext=(0, 10 if agent_idx == 0 else -15),
                                           textcoords="offset points",
                                           ha='center', fontsize=8)
                            except:
                                pass
                
                # Customize plot
                ax.set_yticks([0, 1])
                ax.set_yticklabels([f"Agent 1 ({agent1})", f"Agent 2 ({agent2})"])
                ax.set_xticks(rounds_x)
                ax.set_xlabel("Round")
                ax.set_title("Negotiation Timeline")
                
                # Add final outcome
                agreement = summary_df.iloc[0]['agreement_reached']
                if agreement:
                    ax.text(max_round/2, -0.3, "AGREEMENT REACHED", ha='center', 
                            fontsize=12, color='green', fontweight='bold')
                else:
                    ax.text(max_round/2, -0.3, "NO AGREEMENT", ha='center', 
                            fontsize=12, color='red', fontweight='bold')
                
                # Remove duplicate legend entries
                handles, labels = ax.get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                ax.legend(by_label.values(), by_label.keys(), loc='upper center', 
                          bbox_to_anchor=(0.5, -0.15), ncol=4)
                
                st.pyplot(fig)
                
                # Show raw data
                with st.expander("View Raw Negotiation Data"):
                    st.dataframe(steps_df)
            
            # Show raw summary data
            with st.expander("View Summary Data"):
                st.dataframe(summary_df)
        else:
            st.error("No simulation results found. Check if the simulation ran successfully.")

st.markdown("---")
st.markdown("### 📊 Analysis Tips")
st.markdown("""
- **Successful Negotiations**: Look for simulations where the agents reach an agreement
- **Optimal Outcomes**: The ideal agreement should be close to [0,10] for Agent 1 and [10,0] for Agent 2
- **RL Performance**: Compare the RL agent against other strategies to evaluate its learning
""")