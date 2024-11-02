import gym
import numpy as np

class BlackjackAgent:
    def __init__(self,
                 num_episodes:int = 10000, 
                 gamma: float = 1.0, 
                 epsilon: float = 0.1) -> None:
        
        self.env = gym.make("Blackjack-v1")
        self.num_episodes = num_episodes
        self.gamma = gamma
        self.epsilon = epsilon
        
        self.Q = {}
        self.returns_sum = {}
        self.returns_count = {}

    def epsilon_greedy_policy(self, 
                              state: tuple) -> np.array:
        """Epsilon-Greedy policy based on Q values."""
        num_actions = self.env.action_space.n
        A = np.ones(num_actions) * self.epsilon / num_actions
        best_action = np.argmax(self.Q.get(state, np.zeros(num_actions)))
        A[best_action] += (1.0 - self.epsilon)
        return A

    def play_episode(self) -> list:
        """Play a single episode of Blackjack and update Q-values."""
        episode = []
        state, _ = self.env.reset()
        print(state)
        
        for _ in range(100):
            probs = self.epsilon_greedy_policy(state)
            action = np.random.choice(np.arange(len(probs)), p=probs)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            episode.append((state, action, reward))
            
            if terminated or truncated:
                break

            state = next_state

        return episode

    def update_Q_values(self, 
                        episode: int = 10000) -> None:
        """Update Q values based on the returns from an episode."""
        G = 0
        for state, action, reward in reversed(episode):
            G = self.gamma * G + reward
            
            if (state, action) not in self.returns_sum:
                self.returns_sum[(state, action)] = 0.0
                self.returns_count[(state, action)] = 0.0
            
            self.returns_sum[(state, action)] += G
            self.returns_count[(state, action)] += 1.0
            self.Q[(state, action)] = self.returns_sum[(state, action)] / self.returns_count[(state, action)]

    def train(self) -> None:
        """Train the agent over a specified number of episodes."""
        for _ in range(self.num_episodes):
            episode = self.play_episode()
            self.update_Q_values(episode)

    def get_policy(self) -> dict:
        """Extract the optimal policy from the Q-values."""
        policy = {
            state: np.argmax([self.Q.get((state, action), 0) for action in range(self.env.action_space.n)])
            for state in self.Q
        }
        return policy

    def close(self) -> None:
        """Close the environment."""
        self.env.close()


if __name__ == "__main__":
    agent = BlackjackAgent(num_episodes=10000, gamma=1.0, epsilon=0.1)
    agent.train()
    policy = agent.get_policy()
    print(policy)
    agent.close()
