

# 1. L2-Regularization
## 1.1 Step I Cost Function
$J = J + \frac{\lambda}{m} \sum_1^L (w_l)^2$

```python
# ===== L2 regularization =====
def cost_function_with_regularization (A, y, params, lambd):
    J = cost_function(A, y)
    m = y.shape[1]
    # ===== start =====
    # (lambda / 2m) * np.sum(weights^2)
    W1, W2, W3 = params['W1'], params['W2'], params['W3']
    L2_regularization_cost = lambd * (1 / 2 * m) * (np.sum(np.square(W1)) + np.sum(np.square(W2)) + np.sum(np.square(W2)))
    # ===== end =====

    return J + L2_regularization_cost
```

## 1.2 Step II Backward Propagation

$dJ/dW_{layer} = dJ/dW_{layer} + \frac{\lambda}{m} * W_{layer}$
```python
# ===== L2 regularization =====
def backward_propagation_with_regularization (A, y, params, lambd):
    # dJ_dW = dJ_dW + (lambd / m) * W 
    m = y.shape[1]
    # dJ_dZ3 = A3 - y
    dJ_dZ3 = A['A3'] - y
    """
    Before: dJ_dW3 = (1 / m) * np.dot(A['A2'], dJ_dZ3.T)
    After:  dJ_dW3 = (1 / m) * np.dot(A['A2'], dJ_dZ3.T) + (lambd / m) * params['W3'] 
    """
    # dJ_dW = dJ_dW + (lambd / m) * W 
    dJ_dW3 = (1 / m) * np.dot(A['A2'], dJ_dZ3.T) + (lambd / m) * params['W3'] 
    dJ_db3 = (1 / m) * np.sum(dJ_dZ3, axis = 1, keepdims = True)
    
    # dJ_dZ2
    dJ_dZ2 = np.dot(params['W3'], dJ_dZ3) * ReLU_derivative(A['A2'])
    dJ_dW2 = (1 / m) * np.dot(A['A1'], dJ_dZ2.T) + (lambd / m) * params['W2'] # dJ_dW = dJ_dW + (lambd / m) * W 
    dJ_db2 = (1 / m) * np.sum(dJ_dZ2, axis = 1, keepdims = True)
    
    # dJ_dZ1
    dJ_dZ1 = np.dot(params['W2'], dJ_dZ2) * ReLU_derivative(A['A1'])
    dJ_dW1 = (1 / m) * np.dot(A['A0'], dJ_dZ1.T) + (lambd / m) * params['W1'] # dJ_dW = dJ_dW + (lambd / m) * W 
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
```

# 2. Dropout-Regularization
## 2.1 Step I Forward Propagation
1. A1 (m * n matrix)
2. D1 (m * n matrix) 根据A1的shape尺寸, 建立一个从0-1的随机矩阵
3. D1 = D1 < keep_prob 保留一部分内容
4. A1 = A1 * D1 保存进保留的节点
5. A1 = A1 / keep_prob 拉伸值

```python
# ===== Dropout =====
def forward_propagation_with_dropout (X, params, keep_prob):
    W1, W2, W3 = params['W1'], params['W2'], params['W3']
    b1, b2, b3 = params['b1'], params['b2'], params['b3']
    # Layer 1
    # 1
    A1 = ReLU(np.dot(W1.T, X) + b1)
    # 2
    D1 = np.random.rand(A1.shape[0], A1.shape[1])
    # 3
    D1 = D1 < keep_prob
    # 4
    A1 = A1 * D1
    # 5
    A1 = A1 / keep_prob

    # Layer 2
    A2 = ReLU(np.dot(W2.T, A1) + b2)
    
    D2 = np.random.rand(A2.shape[0], A2.shape[1])
    D2 = D2 < keep_prob
    A2 = A2 * D2
    A2 = A2 / keep_prob
    
    # Layer 3
    # 最后一层不需要处理
    A3 = sigmoid(np.dot(W3.T, A2) + b3)
    
    A = {
        'A0': X,
        'A1': A1,
        'A2': A2,
        'A3': A3,
        'D1': D1, # 需要删减的节点
        'D2': D2 # 需要删减的节点
    }

    return A
```

## 2.2 Step II Backward Propagation
1. dJ_dA2: np.dot(params['W3'], dJ_dZ3)
2. 只保留某些nodes: dJ_dA2 * D2
3. 保留下nodes扩张值: dJ_dA2 / keep_prob
4. dJ_dZ2 = dJ_dA2 * (A2)'

```python
# ===== dropout =====
def backward_propagation_with_dropout (A, y, params, keep_prob):
    D1 = A['D1']
    D2 = A['D2']
    
    m = y.shape[1]
    # ===== dJ_dZ3 = A3 - y =====
    # 最后一层不处理
    dJ_dZ3 = A['A3'] - y
    dJ_dW3 = (1 / m) * np.dot(A['A2'], dJ_dZ3.T)
    dJ_db3 = (1 / m) * np.sum(dJ_dZ3, axis = 1, keepdims = True)
    
    # ===== dJ_dZ2 =====
    # 原来: dJ_dZ2 = np.dot(params['W3'], dJ_dZ3) * ReLU_derivative(A['A2'])
    
    # -------- start -------- 
    # 1. dJ_dA2
    dJ_dA2 = np.dot(params['W3'], dJ_dZ3) 
    
    # 2. dJ_dA2 * D2 只处理保留的结点
    dJ_dA2 = dJ_dA2 * D2 

    # 3. dJ_dA2 / keep_prob 值伸缩
    dJ_dA2 = dJ_dA2 / keep_prob 
    
    # 4. dJ_dZ2
    dJ_dZ2 = dJ_dA2* ReLU_derivative(A['A2']) 
    # -------- end --------
    
    dJ_dW2 = (1 / m) * np.dot(A['A1'], dJ_dZ2.T)
    dJ_db2 = (1 / m) * np.sum(dJ_dZ2, axis = 1, keepdims = True)
    

    # ===== dJ_dZ1 =====
    # 原来 dJ_dZ1 = np.dot(params['W2'], dJ_dZ2) * ReLU_derivative(A['A1'])
    
    # -------- start --------
    dJ_dA1 = np.dot(params['W2'], dJ_dZ2)
    dJ_dA1 = dJ_dA1 * D1
    dJ_dA1 = dJ_dA1 / keep_prob
    dJ_dZ1 = dJ_dA1 * ReLU_derivative(A['A1'])
    # -------- end --------

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
```