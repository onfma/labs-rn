import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import flappy_bird_gymnasium
import gymnasium

env = gymnasium.make("FlappyBird-v0", render_mode="human")

class QNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()
        self.fc = nn.Linear(input_size, 128)
        self.relu = nn.ReLU()
        self.output = nn.Linear(128, output_size)

    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        return self.output(x)

class QLearningAgent:
    def __init__(self, state_size, action_size, gamma=0.99):
        self.q_network = QNetwork(state_size, action_size)
        self.target_network = QNetwork(state_size, action_size)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()

        self.gamma = gamma

    def select_action(self, state):
        q_values = self.q_network(state)
        player_position = state[0][9].item()
        last_top_pipe_position = state[0][1].item()
        last_bottom_pipe_position = state[0][2].item()

        # if player_position < 0.1: 
        #     return 0  # NU sari
        # else:
        #     if player_position > last_top_pipe_position:
        #         print("last_top_pipe_position")
        #         return 1 # sari
        #     if player_position < last_bottom_pipe_position:
        #         print("last_bottom_pipe_position")
        #         return 0  # NU sari
        return torch.argmax(q_values).item()  # Exploatare
                
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
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
epsilon = 0.1
episodes = 1000

agent = QLearningAgent(state_size, action_size)

for episode in range(episodes):
    obs, _ = env.reset()
    state = torch.tensor(obs, dtype=torch.float32).view(1, -1) / 255.0
    total_reward = 0

    while True:
        if np.random.rand() < epsilon:
            action = np.random.choice([0, 1])
        else:
            action = agent.select_action(state)
        next_state, reward, done, _, score = env.step(action)
        next_state = torch.tensor(next_state, dtype=torch.float32).view(1, -1) /255.0
        reward = 0.1 if not done else -1.0 if done and total_reward < 1.0 else 1.0

        agent.update_q_network(state, action, reward, next_state, done)

        total_reward += reward
        state = next_state

        if done:
            agent.update_target_network()
            print(f"Episode: {episode + 1}, Total Reward: {total_reward}, Score: {score}")
            break

env.close()