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
        self.epsilon = epsilon # chance of random action being selected instead of action that is in Q. That helps with a bit more exploration
        
        self.Q = {} # Representation of agent knowledge
        self.returns_sum = {}
        self.returns_count = {}

    def epsilon_greedy_policy(self, 
                              state: tuple) -> np.array:
        """Epsilon-Greedy policy based on Q values."""
        num_actions = self.env.action_space.n
        action_probs = np.ones(num_actions) * self.epsilon / num_actions
        best_action = np.argmax(self.Q.get(state, np.zeros(num_actions)))
        action_probs[best_action] += (1.0 - self.epsilon)
        return action_probs

    def play_episode(self) -> list:
        """Play a single episode of Blackjack and update Q-values."""
        episode = []
        state, _ = self.env.reset()
        
        for _ in range(100):
            probs = self.epsilon_greedy_policy(state)
            action = np.random.choice(np.arange(len(probs)), p=probs) # exploration or exploitation
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

        for state, action, reward in reversed(episode): # we want our agent not to only look at the present action but look at the past actions
            G = self.gamma * G + reward

            if (state, action) not in self.returns_sum:
                self.returns_sum[(state, action)] = 0.0
                self.returns_count[(state, action)] = 0.0
            
            self.returns_sum[(state, action)] += G
            self.returns_count[(state, action)] += 1.0

            self.Q[(state, action)] = self.returns_sum[(state, action)] / self.returns_count[(state, action)]

    def train(self) -> None:
        """Train the agent over a specified number of episodes."""
        for episode_num in range(self.num_episodes):
            if episode_num > self.num_episodes * 0.8:
                self.epsilon = 0.01
            episode = self.play_episode()
            self.update_Q_values(episode)

    def get_policy(self) -> dict:
        """Extract the optimal policy from the Q-values."""
        policy = {
            state: np.argmax([self.Q.get((state, action), 0) for action in range(self.env.action_space.n)])
            for state in self.Q
        }
        return policy
    
    def make_predictions(self, state: tuple) -> int:
        """Makes custom predictions based on what the model have learned.
        
        Args:
            state: A tuple representing the state in the game -> (player_sum: int, dealer_card: int, usable_ace: bool)
        
        Returns:
            int: The index of the action to take based on the learned policy. (0 - Hit, 1 - Stand)
        """

        q_values = [self.Q.get((state, action), 0) for action in range(self.env.action_space.n)]

        action = np.argmax(q_values)
        return action

    def close(self) -> None:
        """Close the environment."""
        self.env.close()
