# 🤝 AI Negotiation Bots

This project explores multi-agent negotiation using autonomous agents with varying strategies — including RL-trained negotiators, cooperative bots, and adversarial agents. Simulations evaluate the ability to reach optimal agreements under dynamic utility conditions.

---

## 🎯 Objectives
- Build negotiation agents with diverse behaviors
- Train RL agents to maximize reward under uncertainty
- Simulate negotiations with multiple utility models
- Analyze results to find equilibrium and social welfare impacts

---

## 🧠 Agent Types
| Agent Type         | Strategy                      |
|--------------------|-------------------------------|
| RuleBasedAgent     | Fixed offer thresholds        |
| RLNegotiatorAgent  | PPO agent with utility maximization |
| CooperativeAgent   | Fair deal seeker              |
| AdversarialAgent   | Maximize gain, risk failure   |

---

## 🚀 Run Simulation

```bash
# Install requirements
pip install -r requirements.txt

# Simulate a negotiation round
python simulations/agent_vs_agent.py --agent1 rl --agent2 adversarial