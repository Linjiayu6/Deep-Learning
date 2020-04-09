import numpy as np
import matplotlib.pyplot as plt

import sklearn
import sklearn.datasets

from tools.activation_function import sigmoid, sigmoid_derivative, ReLU, ReLU_derivative, tanh, tanh_derivative


def load_dataset_moons ():
    # training set
    np.random.seed(1)
    train_X, train_Y = sklearn.datasets.make_moons(n_samples=2000, noise=0.2)
    # testing set
    np.random.seed(2)
    test_X, test_Y = sklearn.datasets.make_circles(n_samples=100, noise=.5)
    # plot
    plt.scatter(train_X[:, 0], train_X[:, 1], c=train_Y, s=40, cmap=plt.cm.Spectral);
    # handle data
    train_X, test_X = train_X.T, test_X.T
    train_Y = train_Y.reshape((1, train_Y.shape[0]))
    test_Y = test_Y.reshape((1, test_Y.shape[0]))
    return train_X, train_Y, test_X, test_Y

def init_parameters_he (Layers):
    np.random.seed(0)
    parameters = {}
    for i in range(1, len(Layers)):
        # 缩放因子
        he = np.sqrt(2. / Layers[i - 1])
        parameters['W' + str(i)] = np.random.randn(Layers[i - 1], Layers[i]) * he
        parameters['b' + str(i)] = np.zeros(shape = (Layers[i], 1))
    return parameters

def forward_propagation (X, params):
    # 3 Layers (X -> A1(relu) -> A2(relu) -> A3(sigmoid))
    W1 = params['W1']
    W2 = params['W2']
    W3 = params['W3']
    
    b1 = params['b1']
    b2 = params['b2']
    b3 = params['b3']
    
    # Layer 1
    A1 = ReLU(np.dot(W1.T, X) + b1)
    # Layer 2
    A2 = ReLU(np.dot(W2.T, A1) + b2)
    # Layer 3
    A3 = sigmoid(np.dot(W3.T, A2) + b3)

    A = {
        'A0': X,
        'A1': A1,
        'A2': A2,
        'A3': A3
    }

    return A

def cost_function (A, y):
    A3 = A['A3']
    m = y.shape[1]
    loss = -y * np.log(A3 + 1e-7) - (1 - y) * np.log(1 - A3 + 1e-7)
    J = (1. / m) * np.nansum(loss)

    return J

def backward_propagation (A, y, params):
    m = y.shape[1]
    # dJ_dZ3 = A3 - y
    dJ_dZ3 = A['A3'] - y
    dJ_dW3 = (1 / m) * np.dot(A['A2'], dJ_dZ3.T)
    dJ_db3 = (1 / m) * np.sum(dJ_dZ3, axis = 1, keepdims = True)
    
    # dJ_dZ2
    dJ_dZ2 = np.dot(params['W3'], dJ_dZ3) * ReLU_derivative(A['A2'])
    dJ_dW2 = (1 / m) * np.dot(A['A1'], dJ_dZ2.T)
    dJ_db2 = (1 / m) * np.sum(dJ_dZ2, axis = 1, keepdims = True)
    
    # dJ_dZ1
    dJ_dZ1 = np.dot(params['W2'], dJ_dZ2) * ReLU_derivative(A['A1'])
    dJ_dW1 = (1 / m) * np.dot(A['A0'], dJ_dZ1.T)
    dJ_db1 = (1 / m) * np.sum(dJ_dZ1, axis = 1, keepdims = True)
    
    grads = {
        'dJ_dW3': dJ_dW3,
        'dJ_db3': dJ_db3,
        'dJ_dW2': dJ_dW2,
        'dJ_db2': dJ_db2,
        'dJ_dW1': dJ_dW1,
        'dJ_db1': dJ_db1
    }
    
    return grads

def update_derivatives (params, grads, alpha, L_len):
    for l in range(1, L_len):
        params['W' + str(l)] -= alpha * grads['dJ_dW' + str(l)]
        params['b' + str(l)] -= alpha * grads['dJ_db'+ str(l)]
    
    return params

def plot_cost (J_arr, alpha):
    plt.plot(J_arr)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(alpha))
    plt.show()
    
def predict (X, y, params, text = '精确度'):
    A = forward_propagation(X, params)
    A3 = A['A3']
    m = A3.shape[1]
    
    y_predict = np.round(A3)
    
    # 精确度预测
    accuracy = (1 - np.sum(abs(y_predict - y)) / m) * 100
    print(text)
    print("===== Accuracy: " + str(accuracy) + '% =====')
  
def predict_dec (params, X):
    A = forward_propagation(X, params)
    A3 = A['A3']
    return np.round(A3)

def plot_decision_boundary(model, X, y):
    axes = plt.gca()
    axes.set_xlim([-1.5, 2.5])
    axes.set_ylim([-1, 1.5])
    
    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=y.ravel(), cmap=plt.cm.Spectral)
    plt.show()