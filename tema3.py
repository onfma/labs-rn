import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium
import flappy_bird_gymnasium
import matplotlib.pyplot as plt


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
    def __init__(self):
        self.state_size = 12
        self.action_size = 2
        self.lr = 0.0001
        
        self.gamma = 0.9999
        self.epsilon = 1.0
        self.eps_decay = 0.999
        self.eps_min = 0.001
        
        
        self.q_network = QNetwork(self.state_size, self.action_size)
        self.target_q_network = QNetwork(self.state_size, self.action_size)
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.target_q_network.eval()
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def policy(self, state):
        # agent.epsilon = max(agent.epsilon * agent.eps_decay, agent.eps_min)
        if np.random.rand() < self.epsilon:
            # print("random")
            return random.choices([0, 1], weights=[90, 10])[0]
        else:
            # print("alg")
            q_values = self.q_network(torch.FloatTensor(state))
            return torch.argmax(q_values).item()

    def train(self, state, action, reward, next_state):
        state_tensor = torch.FloatTensor(state).view(1, -1) / 255.0
        action_tensor = torch.LongTensor([action])
        reward_tensor = torch.FloatTensor([reward]).view(1, 1)
        next_state_tensor = torch.FloatTensor(next_state).view(1, -1) / 255.0

        q_values = self.q_network(state_tensor)
        next_q_values = self.target_q_network(next_state_tensor).detach()
        target_q_values = reward_tensor + self.gamma * torch.max(next_q_values, dim=1, keepdim=True).values
        selected_q_values = torch.gather(q_values, 1, action_tensor.view(-1, 1))

        loss = self.criterion(selected_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

if __name__ == "__main__":
    agent = QLearningAgent()
    episode_scores = []

    episodes = 2000
    for episode in range(episodes):
        env = gymnasium.make("FlappyBird-v0")
        env.reset()
        old_score = 0
        state = env.step(1)[0]

        max_steps = 10000
        for _ in range(max_steps):
            action = agent.policy(state)
            obs, reward, done, _, info = env.step(action)
            agent.train(state, action, reward, obs)
            # env.render()
            state = obs

            if done:
                break
        
        agent.epsilon = max(agent.epsilon * agent.eps_decay, agent.eps_min)
        print("Episode: " + str(episode) + ", Score: " + str(info["score"]) + ", Epsilon: " + str(agent.epsilon))
        episode_scores.append(info["score"])

    plt.plot(range(episodes), episode_scores, label='Episode Scores')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.legend()
    plt.show()
