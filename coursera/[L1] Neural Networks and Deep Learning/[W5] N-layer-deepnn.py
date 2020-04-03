# -*- coding: utf-8 -*
import numpy as np
import h5py
from tools.activation_function import sigmoid, sigmoid_derivative, ReLU, ReLU_derivative, tanh, tanh_derivative

# Loading the data
def load_dataset():  
    train_dataset = h5py.File('data/train_catvnoncat.h5', "r")
    # 209 samples, 64 * 64 pixels
    train_X = np.array(train_dataset["train_set_x"][:]) # (209, 64, 64, 3) 
    train_y = np.array(train_dataset["train_set_y"][:]) # (209,)
    test_dataset = h5py.File('data/test_catvnoncat.h5', "r")  
    # 50 samples
    test_X = np.array(test_dataset["test_set_x"][:]) # (50, 64, 64, 3)
    test_y = np.array(test_dataset["test_set_y"][:]) # (50,)
    # label
    classes = np.array(test_dataset["list_classes"][:]) # [b'non-cat' b'cat'] 
    train_y = np.array([train_y]) # train_y = train_y.reshape((1, train_y.shape[0]))
    test_y = np.array([test_y]) # test_y = test_y.reshape((1, test_y.shape[0])) 
    
    return train_X, train_y, test_X, test_y, classes

train_X, train_y, test_X, test_y, classes = load_dataset()

def handle_data ():
    m_train = train_y.shape[1]
    m_test = test_y.shape[1]
    # train_X: (209, 64, 64, 3) => (209, 64 * 64 * 3) => (64 * 64 * 3, 209)
    X_train_flatten = train_X.reshape(m_train, -1).T
    X_test_flatten = test_X.reshape(m_test, -1).T
    
    # Standardize data to have feature values between 0 and 1.
    X_train_flatten = X_train_flatten / 255.0
    X_test_flatten = X_test_flatten / 255.0

    return X_train_flatten, X_test_flatten

X_train_flatten, X_test_flatten = handle_data()

class deepNN ():
    def __init__ (self, Layers, X, y):
        # data
        self.X = X
        self.y = y

        # ===== parameters =====
        self.Layers = Layers
        self.L = len(Layers)
        self.m = X.shape[1]

        # weights & bias
        W, b = self.init_parameters()
        self.W = W
        self.b = b

        # activations
        self.A = { 'A0': self.X }
        # cost function
        self.J = 0
        # derivatives
        self.dJ = {}

        # ===== hyperparameters =====
        self.alpha = 0.1
        self.interations = 100

        np.random.seed(1)

    def init_parameters (self):
        W, b = {}, {}
        for i in range(1, len(self.Layers)):
            prev_i, next_i = self.Layers[i - 1], self.Layers[i]
            W['W' + str(i)] = np.random.randn(prev_i, next_i) * 0.01
            b['b' + str(i)] = np.zeros(shape = (next_i, 1))
        return W, b

    def forward_propagation (self):
        for i in range(1, self.L):
            A_prev = self.A['A' + str(i - 1)]
            W, b = self.W['W' + str(i)], self.b['b' + str(i)]

            Z = np.dot(W.T, A_prev) + b
            A = ReLU(Z) # 正确情况使用ReLU

            # 最后一项为1或2个结点的输出, 用sigmoid
            if ((i == self.L - 1) and self.Layers[i] == 1):
                A = sigmoid(Z)

            self.A['A' + str(i)] = A

    def cost_function (self):
        A = self.A['A'+ str(self.L - 1)]
        loss = -self.y * np.log(A) - (1 - self.y) * np.log(1 - A)
        self.J = np.sum(loss) / self.m
        print(self.J)

    def get_dJ_dW_db (self, m, A_prev, dJ_dZ):
        dJ_dW = np.dot(A_prev, dJ_dZ.T) / m
        dJ_db = np.sum(dJ_dZ, axis = 1, keepdims = True) / m
        return dJ_dW, dJ_db

    def get_dJ_dZ (self, i):
        # 如果不是最后一层
        W_next = self.W['W' + str(i + 1)]
        dJ_dZ_next = self.dJ['dJ_dZ' + str(i + 1)]
        A = self.A['A'+ str(i)]

        dJ_dA = np.dot(W_next, dJ_dZ_next)
        dA_dZ = ReLU_derivative(A)
        dJ_dZ = dJ_dA * dA_dZ
        return dJ_dZ

    def back_propagation (self):
        m = self.m
        for i in reversed(range(1, self.L)):
            # ===== dJ_dZ =====
            dJ_dZ = []
            if i == self.L - 1: # 最后一个特殊处理
                dJ_dZ = self.A['A'+ str(i)] - self.y
            else:
                dJ_dZ = self.get_dJ_dZ(i)
            self.dJ['dJ_dZ' + str(i)] = dJ_dZ
            
            # ===== dJ_dW, dJ_db =====
            A_prev = self.A['A'+ str(i - 1)]
            dJ_dW, dJ_db = self.get_dJ_dW_db(m, A_prev, dJ_dZ)
            self.dJ['dJ_dW' + str(i)] = dJ_dW
            self.dJ['dJ_db' + str(i)] = dJ_db

    def gradient_descent (self):
        for i in reversed(range(1, len(self.Layers))):
            self.W['W' + str(i)] = self.W['W' + str(i)] - self.alpha * self.dJ['dJ_dW' + str(i)]
            self.b['b' + str(i)] = self.b['b' + str(i)] - self.alpha * self.dJ['dJ_db' + str(i)]

    def train (self):
        for i in range(self.interations):
            self.forward_propagation()
            self.cost_function()
            self.back_propagation()
            self.gradient_descent()
        
        self.cost_function()
        # print('代价函数值', self.A)
        print('代价函数值', self.J)

# data
X, y = X_train_flatten, train_y
n = X.shape[0]

# start
NN = deepNN([n, 7, 1], X, y)
NN.train()