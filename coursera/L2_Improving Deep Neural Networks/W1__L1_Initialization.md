
# He Initialization

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
    