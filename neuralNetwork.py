import numpy as np
import matplotlib.pylab as plt

def step_function(x):
    return np.array(x > 0, dtype=np.int)


'''
シグモイド関数やステップ関数、ReLU関数は非線形関数
'''

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def show_step():
    x = np.arange(-5.0, 5.0, 0.1)
    y = step_function(x)
    plt.plot(x, y)
    plt.ylim(-0.1, 1.1)
    plt.show()

def relu(x):
    return np.maximum(0, x)

def show_sigmoid():
    x = np.arange(-5.0, 5.0, 0.1)
    y = sigmoid(x)
    plt.plot(x, y)
    plt.ylim(-0.1, 1.1)
    plt.show()

def neuralNetwork():
    X = np.array([1.0, 0.5])
    W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    B1 = np.array([0.1, 0.2, 0.3])
    print(W1.shape)
    print(X.shape)
    print(B1.shape)

    A1 = np.dot(X, W1) + B1
    Z1 = sigmoid(A1)
    print(A1)
    print(Z1)

def main():
    neuralNetwork()

main()
