# FNN Builder

A simple class to build feedforward neural networks

### Usage

```python
model = FNN(inputs = 4)

model.addLayer(4, Activation.RELU)
model.addLayer(3, Activation.SIGMOID)
model.addLayer(3, Activation.SOFTMAX)

model.train(x_train, y_train, epochs = 10000, rate = .01, batch_size = 5)
```

### Activation functions
- Sigmoïd
- ReLU
- Softmax (only for last layer with Cross-Entropy)

### Loss functions
- Categorical Cross Entropy (with softmax last layer)
- Mean Squared Error [TODO]

### Optimization
- Stochastic gradient descent
- Mini-batch
- Batch
