import numpy as np
from matplotlib import pyplot as plt


def display_random_digit(x, y):
    rows = x.shape[0]
    random_n = np.random.randint(rows, size=1)
    image = x[random_n]
    digit = y[random_n][0]

    image = np.reshape(image, (20, 20)).T
    fig, ax = plt.subplots()
    ax.imshow(image, interpolation='nearest')
    ax.set_title(f'Dígito {digit}')
    plt.show()


def sigmoid_gradient(z):
    g = sigmoid(z) * (1 - sigmoid(z))
    return g


def sigmoid(z):
    g = np.power((1.0 + np.exp(-z)), -1)
    return g


class FeedforwardNeuralNetwork:

    def __init__(self, input_layer, hidden_layer, output_layer):
        self.output_layer = output_layer
        self.theta1 = np.random.uniform(-1, 1, (hidden_layer, input_layer + 1))
        self.theta2 = np.random.uniform(-1, 1, (output_layer, hidden_layer + 1))

    def get_encoded_labels(self, x, y):
        m = x.shape[0]
        encoded_labels = np.zeros((m, self.output_layer), dtype=int)
        encoded_labels[range(encoded_labels.shape[0]), y] = 1
        return encoded_labels

    def train(self, x, y, lambd, epoch=50, learning_rate=0.2):
        encoded_labels = self.get_encoded_labels(x, y)

        for i in range(epoch):
            j, theta1_grad, theta2_grad = self.cost_function(x, encoded_labels, lambd)
            self.theta1 -= learning_rate*theta1_grad
            self.theta2 -= learning_rate*theta2_grad

    def cost_function(self, x, encoded_labels, lambd):
        m = x.shape[0]
        a1 = np.hstack([np.ones((m, 1)), x])
        z2 = a1 @ self.theta1.T
        a2 = sigmoid(z2)
        a2 = np.hstack([np.ones((a2.shape[0], 1)), a2])
        z3 = a2 @ self.theta2.T
        a3 = sigmoid(z3)
        hypothesis = a3
        regularization = 0
        if lambd != 0:
            regularization = -1 * np.sum(np.square(self.theta1[:, 0])) - np.sum(np.square(self.theta2[:, 0]))
            regularization += + np.sum(np.square(self.theta1)) + np.sum(np.square(self.theta2))
            regularization *= lambd / (2 * m)

        j = np.trace(encoded_labels.T @ np.log(hypothesis)) + np.trace((1 - encoded_labels.T) @ np.log(1 - hypothesis))
        j *= (-1 / m)
        j += regularization

        # Backpropagation
        d3 = a3 - encoded_labels
        d2 = (d3 @ self.theta2[:, 1:]) * sigmoid_gradient(z2)

        delta1 = (1 / m) * (d2.T @ a1)
        delta2 = (1 / m) * (d3.T @ a2)

        temp_theta1 = np.copy(self.theta1)
        temp_theta1[:, 0] = 0
        temp_theta2 = np.copy(self.theta2)
        temp_theta2[:, 0] = 0

        theta1_grad = delta1 + (lambd / m) * temp_theta1
        theta2_grad = delta2 + (lambd / m) * temp_theta2
        return j, theta1_grad, theta2_grad

    def predict(self, x):
        m = x.shape[0]
        h1 = sigmoid(np.hstack([np.ones((m, 1)), x]) @ self.theta1.T)
        h2 = sigmoid(np.hstack([np.ones((m, 1)), h1]) @ self.theta2.T)
        h = np.argmax(h2, axis=1)
        return h

    def get_accuracy(self, x, y):
        h0 = self.predict(x)
        return np.mean(h0 == y) * 100
