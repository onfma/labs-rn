import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import flappy_bird_gymnasium
import gymnasium

class QNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()
        self.reshape = nn.Flatten()
        self.fc1 = nn.Linear(input_size, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, output_size)

    def forward(self, x):
        x = self.reshape(x)
        x = self.fc1(x)
        x = self.relu(x)
        return self.fc2(x)

class QLearningAgent:
    def __init__(self, state_size, action_size, gamma=0.99, epsilon=1.0, epsilon_decay=0.99, epsilon_min=0.01):
        self.q_network = QNetwork(state_size, action_size)
        self.target_network = QNetwork(state_size, action_size)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.0001)
        self.loss_fn = nn.MSELoss()

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

    def select_action(self, state):
        if torch.rand(1).item() < self.epsilon:
            return torch.randint(0, 2, (1,)).item()  # Explore
        else:
            with torch.no_grad():
                q_values = self.q_network(state)
                return torch.argmax(q_values).item()  # Exploit

    def update_q_network(self, state, action, reward, next_state, done):
        self.optimizer.zero_grad()

        q_values = self.q_network(state)
        target_q_values = q_values.clone().detach()

        if done:
            target_q_values[0][action] = reward
        else:
            with torch.no_grad():
                target_q_values[0][action] = reward + self.gamma * torch.max(self.target_network(next_state))

        loss = self.loss_fn(q_values, target_q_values)
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

# Hyperparametri
state_size = 12
action_size = 2
episodes = 1000

env = gymnasium.make("FlappyBird-v0", render_mode="human")

agent = QLearningAgent(state_size, action_size)

for episode in range(episodes):
    obs, _ = env.reset()
    state = torch.tensor(obs, dtype=torch.float32).view(1, -1) / 255.0
    total_reward = 0

    while True:
        action = agent.select_action(state)
        next_state, reward, done, _, score = env.step(action)
        next_state = torch.tensor(next_state, dtype=torch.float32).view(1, -1) /255.0
        reward = 0.1 if not done else -1.0 if done and total_reward < 1.0 else 1.0

        agent.update_q_network(state, action, reward, next_state, done)

        total_reward += reward
        state = next_state

        if done:
            agent.update_target_network()
            agent.epsilon = max(agent.epsilon * agent.epsilon_decay, agent.epsilon_min)
            print(f"Episode: {episode + 1}, Epsilon: {agent.epsilon}, Total Reward: {total_reward}, Score: {score}")
            break

env.close()