# BlackjackAgent 🎲 - Reinforcement Learning Agent for Blackjack

A reinforcement learning agent trained to play Blackjack using Monte Carlo control and an epsilon-greedy policy. This project leverages OpenAI Gym's **Blackjack-v1** environment, allowing the agent to learn and improve its strategy through repeated episodes.

## 🎯 Project Overview

The `BlackjackAgent` is designed to play and learn optimal strategies for Blackjack by:

- **Exploring different actions** and observing their outcomes.
- **Updating Q-values** based on the cumulative returns from entire episodes.
- **Balancing exploration and exploitation** using an epsilon-greedy policy.

## ⚙️ Key Features

- **Epsilon-Greedy Policy**: Balances exploration and exploitation to improve action selection.
- **Monte Carlo Control**: Updates Q-values based on observed returns, helping the agent learn from episodes of gameplay.
- **Adjustable Parameters**: Configurable number of episodes, discount factor (`gamma`), and exploration rate (`epsilon`).
- **Optimal Policy Extraction**: Retrieves the optimal policy for each state based on learned Q-values after training.

## 🚀 Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/BlackjackAgent.git
   cd BlackjackAgent
   ```
2. Install required dependencies

```ps
pip install -r requirements.txt
```

# 📝 Usage

To train the agent and extract the learned policy, run:

```python
from blackjack_agent import BlackjackAgent

# Initialize the agent
agent = BlackjackAgent(num_episodes=10000, gamma=1.0, epsilon=0.1)

# Train the agent
agent.train()

# Get the learned policy
policy = agent.get_policy()
print(policy)

# Close the environment when done
agent.close()
```

# 📂 Code Structure

```python
.
├── blackjack_agent.py        # Main implementation of the BlackjackAgent class
├── README.md                 # Project documentation
└── requirements.txt          # Python dependencies
```

# 📊 Example Output

After training, the agent will output an optimal policy for Blackjack based on the learned Q-values. Here’s an example:

```python
{
    (18, 10, False): 0,  # Action: Stick
    (15, 7, False): 1,   # Action: Hit
    ...
}
```

# 📚 How It Works

## Epsilon-Greedy Policy

The agent selects actions based on the epsilon-greedy policy:

- With probability `epsilon`, it explores a random action.
- Otherwise, it exploits the best-known action for the current state based on Q-values.

## Monte Carlo Q-Value Updates

The agent uses **Monte Carlo control** to update Q-values. Each episode accumulates rewards, and Q-values are adjusted based on the episode's overall return.

# 🔧 Parameters

You can configure the following parameters in `BlackjackAgent`:

- `num_episodes`: Number of episodes to train (default: 10,000).
- `gamma`: Discount factor for future rewards (default: 1.0).
- `epsilon`: Exploration rate for epsilon-greedy policy (default: 0.1).
