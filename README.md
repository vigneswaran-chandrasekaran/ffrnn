# Flipflop Recurrent Neural Network

**Install**

```
pip install ffrnn
```

**Usage**

PyTorch class for flipflop recurrent neural networks

Custom `Pytorch` class for flipflop neural networks

```python
import torch
from ffrnn.fftorch import FF
seq_len = 200
input_size = 1
hidden_size = 4
x = torch.rand(seq_len, 1, input_size)
output = FF(input_size, hidden_size, num_layers=2)
```
---
Tensorflow2.x class for flipflop recurrent neural networks

Custom `TensorFlow 2.x` class for flipflop neural networks

Can be passed into `RNN` wrapper as custom `RNNCell`
```python
import tensorflow as tf
from ffrnn.fftf import FF
tf.keras.layers.RNN(FF(output_dim), input_shape=(seq_len, input_dim))
```
---


**References**

https://github.com/vigneswaran-chandrasekaran/Flipflop_experiments

Kumari, S., Chandrasekaran, V. & Chakravarthy, V.S. The flip-flop neuron: a memory efficient alternative for solving challenging sequence processing and decision-making problems. Neural Comput & Applic 35, 24543–24559 (2023). https://doi.org/10.1007/s00521-023-08552-7
