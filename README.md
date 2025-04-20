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
# Create and activate a virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate

# Install requirements
pip install -r requirements.txt

# Simulate a negotiation round
python simulations/agent_vs_agent.py --agent1 rl --agent2 adversarial
```

---

## ✅ Initial Task Checklist
- [ ] Create negotiation environment with reward signals
- [ ] Implement baseline rule-based agent
- [ ] Train RL-based negotiation agent (PPO/DQN)
- [ ] Log negotiation outcomes for analysis
- [ ] Create visualization notebook

---

## 📁 Folder Overview

| Folder             | Description                              |
|--------------------|------------------------------------------|           
| agents/            | All agent implementations                |
| environment/       | Negotiation gym-style environment        |
| simulations/       | Scripts to run agent-vs-agent scenarios  |
| models/            | RL policy models                         |
| notebooks/         | Strategy exploration & plotting          |

---

## 🧠 Sample Agent Code

```python
class RuleBasedAgent:
    def __init__(self, min_accept=0.6):
        self.min_accept = min_accept
    
    def propose(self, last_offer):
        return 1.0 - (1.0 - self.min_accept) * 0.9
    
    def respond(self, offer):
        return offer >= self.min_accept