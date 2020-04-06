
# Course 2: Improving Deep Neural Networks: Hyperparameter tuning, Regularization and Optimization

```
Week 1 - Initialization
Week 1 - Regularization
Week 1 - Gradient Checking
Week 2 - Optimization Methods
Week 3 - TensorFlow Tutorial
```

***

# Week 1 - Initialization

## Initialization
- W1__Initialization.ipynb

```python
# weights & bias 参数初始化的三种创建对NN影响

1. Zero Initialization (❌)
2. Random Initialization (❌不能选用过大的随机数, 会导致 "梯度爆炸或梯度消失")
3. He Initialization (✅ 增加缩放因子)
```

## He Initialization说明

**乘以缩放因子** 
$W = W * \sqrt{\frac{2}{self.Layers[i - 1]}}$

```python
def init_parameters_he (self):
     parameters = {}
     for i in range(1, len(self.Layers)):
         # 缩放因子
         he = np.sqrt(2. / self.Layers[i - 1])
         # W * 缩放因子
         parameters['W' + str(i)] = np.random.randn(self.Layers[i - 1], self.Layers[i]) * he
         parameters['b' + str(i)] = np.zeros(shape = (self.Layers[i], 1))
     return parameters
```

[R: Initialization](https://github.com/Kulbear/deep-learning-coursera/blob/master/Improving%20Deep%20Neural%20Networks%20Hyperparameter%20tuning%2C%20Regularization%20and%20Optimization/Initialization.ipynb)

## 疑问
```python
# 疑问:
1. 为什么weights 随机数过大会导致[梯度爆炸或梯度消失]?

2. 为什么He Initialization 增加缩放因子会解决问题? 解决什么问题?
```
***


# Week 2 - Regularization

## Regularization
- W2__Regularization.ipynb

```python
# 有无正则化 三种方式的对比
1. Non-regularized
会出现过度拟合情况, training set 正确率就非常的高
2. L2-Regularization ✅
3. Dropout-Regularization ✅
```

## L2-Regularization
### Step I 损失函数
$J = J + \frac{\lambda}{m} \sum_1^L (w_l)^2$

```python
# ===== L2 regularization =====
def cost_function_with_regularization (A, y, params, lambd):
    J = cost_function(A, y)
    m = y.shape[1]
    # ===== L2 regularization =====
    # (lambda / 2m) * np.sum(weights^2)
    W1, W2, W3 = params['W1'], params['W2'], params['W3']
    L2_regularization_cost = lambd * (1 / 2 * m) * (np.sum(np.square(W1)) + np.sum(np.square(W2)) + np.sum(np.square(W2)))

    return J + L2_regularization_cost
```

### Step II 反向传播

$dJ/dW_{layer} = dJ/dW_{layer} + \frac{\lambda}{m} * W_{layer}$
```python
# ===== L2 regularization =====
def backward_propagation_with_regularization (A, y, params, lambd):
    # dJ_dW = dJ_dW + (lambd / m) * W 
    m = y.shape[1]
    # dJ_dZ3 = A3 - y
    dJ_dZ3 = A['A3'] - y
    dJ_dW3 = (1 / m) * np.dot(A['A2'], dJ_dZ3.T) + (lambd / m) * params['W3'] # dJ_dW = dJ_dW + (lambd / m) * W 
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

## Dropout-Regularization
### Step I 正向传播
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
    A1 = ReLU(np.dot(W1.T, X) + b1)
    
    D1 = np.random.rand(A1.shape[0], A1.shape[1])
    D1 = D1 < keep_prob
    A1 = A1 * D1
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

### Step II 反向传播
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
    dJ_dA2 = np.dot(params['W3'], dJ_dZ3) # 1. dJ_dA2
    dJ_dA2 = dJ_dA2 * D2 # 2. dJ_dA2 * D2 只处理保留的结点
    dJ_dA2 = dJ_dA2 / keep_prob # 3. dJ_dA2 / keep_prob 值伸缩   
    dJ_dZ2 = dJ_dA2* ReLU_derivative(A['A2']) # 4. dJ_dZ2
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

[R: Regularization](https://github.com/Kulbear/deep-learning-coursera/blob/master/Improving%20Deep%20Neural%20Networks%20Hyperparameter%20tuning%2C%20Regularization%20and%20Optimization/Regularization.ipynb)


## 疑问
```python
# 疑问:
1. 为什么我的损失函数为负数? 但是并不影响整体操作?
```