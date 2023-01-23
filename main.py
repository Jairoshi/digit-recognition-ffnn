import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from FeedforwardNeuralNetwork import FeedforwardNeuralNetwork, display_random_digit
import matplotlib.pyplot as plt

X = pd.read_csv('X.csv', sep=',', header=None).values
y = pd.read_csv('y.csv', sep=',', header=None).values
y = np.squeeze(y)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
# display_random_digit(X, y)
nn = FeedforwardNeuralNetwork(400, 25, 10)
labels = nn.get_encoded_labels(x_train, y_train)

num_labels = 10
lambd = 3
n = 150
j = np.zeros(n,)
training_accuracy = np.zeros(n,)
test_accuracy = np.zeros(n,)
epoch = 50

# Training
for i in range(0, n):
    j[i] = nn.cost_function(x_train, labels, lambd)[0]
    training_accuracy[i] = nn.get_accuracy(x_train, y_train)
    test_accuracy[i] = nn.get_accuracy(x_test, y_test)
    nn.train(x_train, y_train, lambd, epoch=epoch)

    print(f"Training step {i*epoch}:")
    print("Cost function: %.2f" % j[i])
    print("Training set accuracy: %.2f" % training_accuracy[i], "%")
    print("Test set accuracy: %.2f" % test_accuracy[i], "%")

# Plot results
fig1, ax1 = plt.subplots()
ax1.plot(np.linspace(0, n*epoch, n), training_accuracy, label="Training")
ax1.plot(np.linspace(0, n*epoch, n), test_accuracy, label="Test")
plt.legend(loc="upper left")
ax1.set_title("Precisión")
ax1.set_xlabel("Training")
ax1.set_ylabel("Precisión")

fig2, ax2 = plt.subplots()
ax2.plot(np.linspace(0, n*epoch, n), j)
ax2.set_title("Cost")
ax2.set_xlabel("Training")
ax2.set_ylabel("Cost")
plt.show()