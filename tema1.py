# Antrenati 10 perceptroni pentru identificarea cifrei dintr-o imagine.

# In esenta, vom avea cate un perceptron pentru fiecare cifra, 
# care va invata sa diferentieze intre cifra respectiva si restul cifrelor. 
# Prin asamblarea acestora putem deduce ce cifra ar fi intr-o imagine specifica.

import requests
import pickle
import gzip
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

# descarcare database
url = "https://github.com/mnielsen/neural-networks-and-deep-learning/raw/master/data/mnist.pkl.gz"
with open("mnist.pkl.gz", "wb") as fd:
    fd.write(requests.get(url).content)

# incarcare MNIST dataset
with gzip.open("mnist.pkl.gz", "rb") as fd:
    train_set, valid_set, test_set = pickle.load(fd, encoding="latin")

# separare dataset in features & labels
train_features, train_labels = train_set

# divizare in mini-batches
batch_size = 1000
num_batches = len(train_features) // batch_size

# lista de perceptrons 
perceptrons = []

# antrenare 10 perceptrons, unul pt fiecare cifra
for digit in range(10):
    # Create a binary classification problem (digit vs. non-digit)
    binary_labels = (train_labels == digit).astype(int)

    # creare & antrenare perceptron cu mini-batches training
    perceptron = Perceptron()
    for batch in range(num_batches):
        start = batch * batch_size
        end = (batch + 1) * batch_size
        perceptron.partial_fit(train_features[start:end], binary_labels[start:end], classes=[0, 1])

    perceptrons.append(perceptron)


### TESTARE

# incarcare testset
test_features, test_labels = test_set

# testarea pe training set/test set a perceptronilor antrenati
def evaluate_perceptrons(data_features, data_labels):
    # initializare predictii pt fiecare perceptron
    perceptron_predictions = np.zeros((len(data_features), 10))

    for digit in range(10):
        perceptron = perceptrons[digit]
        perceptron_predictions[:, digit] = perceptron.decision_function(data_features)

    predicted_digits = np.argmax(perceptron_predictions, axis=1)
    accuracy = accuracy_score(data_labels, predicted_digits)
    return accuracy

# evaluarea acuratete training set
train_accuracy = evaluate_perceptrons(train_features, train_labels)
print(f"Accuracy of the perceptron ensemble on the training set: {train_accuracy * 100:.2f}%")

# evaluarea acuratete test set
test_accuracy = evaluate_perceptrons(test_features, test_labels)
print(f"Accuracy of the perceptron ensemble on the test set: {test_accuracy * 100:.2f}%")