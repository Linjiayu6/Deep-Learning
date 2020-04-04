# Loading the data
import numpy as np
import h5py

def load_data():  
    train_dataset = h5py.File('train_catvnoncat.h5', "r")
    # 209 samples, 64 * 64 pixels
    train_X = np.array(train_dataset["train_set_x"][:]) # (209, 64, 64, 3) 
    train_y = np.array(train_dataset["train_set_y"][:]) # (209,)
  
    test_dataset = h5py.File('test_catvnoncat.h5', "r")  
    # 50 samples
    test_X = np.array(test_dataset["test_set_x"][:]) # (50, 64, 64, 3)
    test_y = np.array(test_dataset["test_set_y"][:]) # (50,)
  
    # label
    classes = np.array(test_dataset["list_classes"][:]) # [b'non-cat' b'cat'] 
    
    # y = [...] => y = [[...]]
    train_y = np.array([train_y]) # train_y = train_y.reshape((1, train_y.shape[0]))
    test_y = np.array([test_y]) # test_y = test_y.reshape((1, test_y.shape[0])) 
    return train_X, train_y, test_X, test_y, classes

def on_data (train_X, test_X):
    # Explore ur dataset
    # the number of training samples
    m_train = train_X.shape[1] # 209

    # the number of test samples
    m_test = test_X.shape[1] # 50

    # size of image 64 * 64 * 3
    # num_px = train_X.shape[1: 4] # (64, 64, 3)

    # train_X: (209, 64, 64, 3) => (209, 64 * 64 * 3) => (64 * 64 * 3, 209)
    X_train_flatten = train_X.reshape(m_train, -1).T
    X_test_flatten = test_X.reshape(m_test, -1).T
    
    # Standardize data to have feature values between 0 and 1.
    X_train_flatten = X_train_flatten / 255
    X_test_flatten = X_test_flatten / 255

    return X_train_flatten, X_test_flatten