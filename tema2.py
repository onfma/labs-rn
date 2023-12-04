import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import requests
import gzip
import numpy as np

url = "https://github.com/mnielsen/neural-networks-and-deep-learning/raw/master/data/mnist.pkl.gz"
with open("mnist.pkl.gz", "wb") as fd:
    fd.write(requests.get(url).content)

with gzip.open("mnist.pkl.gz", "rb") as fd:
    train_set, valid_set, test_set = pickle.load(fd, encoding="latin")

train_X, train_y = torch.from_numpy(train_set[0]), torch.from_numpy(train_set[1])
test_X, test_y = torch.from_numpy(test_set[0]), torch.from_numpy(test_set[1])

# initializare params
input_size = 28 * 28  # dim imags MNIST
hidden_size = 128     # nr neuroni strat ascuns
output_size = 10      # nr clase clase (cifre 0 - 9)
num_epochs = 10       # nr epoci
batch_size = 10       # dim batch
learning_rate = 0.1  # rata de inv

# functia ReLU de activare
def relu(x):
    return torch.max(torch.tensor(0, dtype=x.dtype), x)

# functia Softmax pt output
def softmax(x):
    exp_x = torch.exp(x - torch.max(x, dim=1, keepdim=True).values)
    return exp_x / torch.sum(exp_x, dim=1, keepdim=True)

# functia crossentropie pt cost
def cross_entropy_loss(outputs, labels):
    log_softmax_outputs = torch.log_softmax(outputs, dim=1)
    selected_log_softmax = log_softmax_outputs[range(len(outputs)), labels]
    loss = -torch.mean(selected_log_softmax)
    return loss

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = relu(x)
        x = self.fc2(x)
        x = softmax(x)
        return x


model = MLP(input_size, hidden_size, output_size)
optimizer = optim.Adam(model.parameters(), lr=0.01)

# mini-batch-uri
train_dataset = TensorDataset(train_X, train_y.long())
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# antrenare
for epoch in range(num_epochs):
    for inputs, labels in train_loader:
        model.zero_grad()  # reset gradient
        outputs = model(inputs.float()) # forward propag
        loss = cross_entropy_loss(outputs, labels) # calcul cost
        loss.backward() # back propag

        # actualizare ponderi
        with torch.no_grad():
            for param in model.parameters():
                param -= learning_rate * param.grad

# evaluare set testare
with torch.no_grad():
    outputs = model(test_X.float())
    _, predicted = torch.max(outputs, 1)
    predicted_np = predicted.numpy()
    test_y_np = test_y.numpy()

# calcul acuratete
correct = 0
total = 0
for pred, true_label in zip(predicted_np, test_y_np):
    if pred == true_label:
        correct += 1
    total += 1

accuracy = correct / total
print(f"Acuratețe generală pe setul de testare: {accuracy:.2%}")


from sklearn.metrics import f1_score
f1_scores = f1_score(test_y, predicted, average=None)
print("F1 pentru fiecare clasă pe setul de testare:")
for i, f1 in enumerate(f1_scores):
    print(f'Clasa {i}: {f1:.2f}')


# Acuratețe generală pe setul de testare: 97.05%
# F1 pentru fiecare clasă pe setul de testare:
# Clasa 0: 0.98
# Clasa 1: 0.98
# Clasa 2: 0.97
# Clasa 3: 0.97
# Clasa 4: 0.97
# Clasa 5: 0.97
# Clasa 6: 0.97
# Clasa 7: 0.97
# Clasa 8: 0.96
# Clasa 9: 0.96