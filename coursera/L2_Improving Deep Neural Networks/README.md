
# Course 2: Improving Deep Neural Networks: Hyperparameter tuning, Regularization and Optimization

```
Week 1 - Initialization
Week 1 - Regularization
Week 1 - Gradient Checking
Week 2 - Optimization Methods
Week 3 - TensorFlow Tutorial
```

***

# Week 1 - L1 Initialization

## 1.1 Initialization
- W1__L1_Initialization.ipynb

```python
# weights & bias 参数初始化的三种创建对NN影响

1. Zero Initialization (❌)
2. Random Initialization (❌不能选用过大的随机数, 会导致 "梯度爆炸或梯度消失")
3. He Initialization (✅ 增加缩放因子)
```
[R: Initialization](https://github.com/Kulbear/deep-learning-coursera/blob/master/Improving%20Deep%20Neural%20Networks%20Hyperparameter%20tuning%2C%20Regularization%20and%20Optimization/Initialization.ipynb)

## 1.2 He Initialization说明
[W1__L1_Initialization.md](https://github.com/Linjiayu6/Deep-Learning/blob/master/coursera/L2_Improving%20Deep%20Neural%20Networks/W1__L1_Initialization.md)

## 1.3 疑问
```python
# 疑问:
1. 为什么weights 随机数过大会导致[梯度爆炸或梯度消失]?

2. 为什么He Initialization 增加缩放因子会解决问题? 解决什么问题?
```
***


# Week 1 - L2 Regularization

## 2.1 Regularization
- W1__L2_Regularization.ipynb

```python
# 有无正则化 三种方式的对比
1. Non-regularized
会出现过度拟合情况, training set 正确率就非常的高
2. L2-Regularization ✅
3. Dropout-Regularization ✅
```
[R: Regularization](https://github.com/Kulbear/deep-learning-coursera/blob/master/Improving%20Deep%20Neural%20Networks%20Hyperparameter%20tuning%2C%20Regularization%20and%20Optimization/Regularization.ipynb)

## 2.2 L2 & Dropout 说明
[W1__L2_Regularization.md](https://github.com/Linjiayu6/Deep-Learning/blob/master/coursera/L2_Improving%20Deep%20Neural%20Networks/W1__L2_Regularization.md)


## 2.3 疑问
```python
# 疑问:
1. 为什么我的损失函数为负数? 但是并不影响整体操作?
```

***

# Week 1 - L3 Gradient Checking
## 3.1 Gradient Checking
- W1__L3_Gradient_Checking.ipynb

```
TODO: 后面有问题 !!!
```

***

# Week 2 - Optimization Methods

## 4.1 Mini Batch Gradient Descent
- W2__L1_Optimization Methods (Mini-Batch GD).ipynb
每组数量, 最好是2的次幂, eg: $2^6 = 64$

## 4.2 Momentum & Adam
- W2__L2_Optimization Methods (Momentum & Adam).ipynb