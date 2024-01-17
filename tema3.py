import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import flappy_bird_gymnasium
import gymnasium

class QNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

env = gymnasium.make("FlappyBird-v0", render_mode="human")

gamma = 0.99  # discount
epsilon = 0.1  # explorare
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

model = QNetwork(input_size=state_size, output_size=action_size)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Funcția Q-learning
def q_learning_train(model, state, action, reward, next_state, done, scor):
    state = torch.tensor(state, dtype=torch.float32).view(1, -1)
    next_state = torch.tensor(next_state, dtype=torch.float32).view(1, -1)

    Q_values = model(state)
    next_Q_values = model(next_state)
    print(state)

    # Calcularea target-ului Q
    target = Q_values.clone().detach()
    print(target)
    print(target[0, action])
    
    if done:
        target[0, action] = reward
    else:
        if state[0][9] < 0: # pasare in afara ecranului (prea sus) => NU sare
            target[0, 0] = reward + gamma * next_Q_values[0, 0]
        else:
            if next_state[0][9] <= next_state[0][2]: # pasare mai sus de teava de sus => NU sare
                target[0, 0] = reward + gamma * next_Q_values[0, 0]
            else:
                target[0, action] = reward + gamma * torch.max(next_Q_values)


    loss = criterion(Q_values, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


num_episodes = 1000

for episode in range(num_episodes):
    obs, _ = env.reset()
    total_reward = 0

    while True:
        state = obs
        if np.random.rand() < epsilon:
            action = np.random.choice([0, 1])
        else:
            Q_values = model(torch.tensor(state, dtype=torch.float32).view(1, -1))
            action = torch.argmax(Q_values).item()

        next_obs, reward, terminated, _, scor = env.step(action)

        q_learning_train(model, state, action, reward, next_obs, terminated, scor)

        obs = next_obs
        total_reward += reward

        if terminated:
            break

    print(f"Episode: {episode + 1}, Total Reward: {total_reward}")

env.close()
