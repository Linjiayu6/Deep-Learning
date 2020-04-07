import numpy as np
import matplotlib.pyplot as plt

from tools.activation_function import sigmoid, sigmoid_derivative, ReLU, ReLU_derivative, tanh, tanh_derivative

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