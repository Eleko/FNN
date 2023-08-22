# FNN Builder

A simple class to build feedforward neural networks

### Usage

#### Build
```python
model = FNN(inputs = 4)
model.normalize(x = True, y = False)
model.addLayer(neurons = 4, Activation.RELU)
model.addLayer(neurons = 3, Activation.SIGMOID)
model.addLayer(neurons = 3, Activation.SOFTMAX)
```

#### Train
```python
model.train(x_train, y_train, epochs = 10000, rate = .01, batch_size = 5)
```
---
### Activation functions
- Sigmoïd
- ReLU
- Softmax (only for last layer with Cross-Entropy)

### Loss functions
- Categorical Cross Entropy (with softmax last layer)
- Mean Squared Error [TODO]

### Normalization
- Rescaling (min-max normalization)

### Gradient-based Optimization
- Stochastic Gradient Descent : `batch_size = 1`
- Mini-batch : `batch_size = n` 
- Batch :  `batch_size = len(x_train.index)`

---
### Requirements
- numpy
- pandas
- matplotlib
