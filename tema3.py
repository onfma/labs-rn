import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import flappy_bird_gymnasium
import gymnasium

# rn convolutionala
class CNNQNetwork(nn.Module):
    def __init__(self, input_channels, output_size):
        super(CNNQNetwork, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, stride=2)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        self.relu3 = nn.ReLU()
        self.fc1 = nn.Linear(64 * 5 * 5, 256)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(256, output_size)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = x.view(-1, 64 * 5 * 5)
        x = self.relu4(self.fc1(x))
        x = self.fc2(x)
        return x
    
# creare mediu de joc
env = gymnasium.make("FlappyBird-v0", render_mode="human")

gamma = 0.99  # discount
epsilon = 0.1  # explorare
learning_rate = 0.001
input_channels = 3  # RGB
state_size = (input_channels, 84, 84)  # dim img redimensionata
action_size = env.action_space.n

# creare rn
model = CNNQNetwork(input_channels=input_channels, output_size=action_size)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

# Funcția de preprocesare a imaginilor
def preprocess_image(observation):
    image_data, _ = observation
    image = np.array(image_data, dtype=np.float32)
    
    # Redimensionează imaginea la dimensiunile așteptate de rețea
    # image = image.reshape((1, 1, 12, 1))
    image = image.reshape((1, 12, 1))
    image = torch.unsqueeze(torch.tensor(image), 0)

    # Normalizare la intervalul [0, 1]
    image = image / 255.0

    # Convertirea la tensor PyTorch
    image = torch.tensor(image, dtype=torch.float32)
    print(image)
    return image

# Funcția Q-learning cu imagini
def q_learning_train(model, state, action, reward, next_state, done):
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)

    # Obținerea estimării Q pentru starea actuală și viitoare
    Q_values = model(state)
    next_Q_values = model(next_state)

    # Calcularea target-ului Q
    target = Q_values.clone().detach()
    
    if done:
        target[0, action] = reward  # -1.0 pentru moarte
    else:
        # Utilizăm recompensele specifice pentru acest joc
        target[0, action] = reward + gamma * torch.max(next_Q_values)

    # Calcularea pierderii și actualizarea rețelei neuronale
    loss = criterion(Q_values, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Antrenamentul cu imagini
num_episodes = 1000

for episode in range(num_episodes):
    obs = env.reset()
    total_reward = 0

    while True:
        # Alegerea acțiunii utilizând politica epsilon-greedy
        state = preprocess_image(obs)

        if np.random.rand() < epsilon:
            action = np.random.choice([0, 1])
        else:
            Q_values = model(state)
            action = torch.argmax(Q_values).item()

        # Aplicarea acțiunii și obținerea următoarei stării și recompensei
        next_obs, reward, terminated, _, _ = env.step(action)

        next_state = preprocess_image(next_obs)

        # Antrenamentul rețelei neuronale folosind algoritmul Q-learning
        q_learning_train(model, state, action, reward, next_state, terminated)

        obs = next_obs
        total_reward += reward

        if terminated:
            break

    print(f"Episode: {episode + 1}, Total Reward: {total_reward}")

env.close()
