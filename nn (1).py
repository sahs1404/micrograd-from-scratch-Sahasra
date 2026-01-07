from value import Value
import random

class Neuron:
    """
    A single neuron: weights + bias.
    Forward pass: tanh( Σ (w_i * x_i) + b )
    """

    def __init__(self, nin):
        # nin = number of inputs
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(0.0)

    def __call__(self, x):
        # x is a list of inputs (Value objects)
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.tanh()

    def parameters(self):
        # return all trainable parameters
        return self.w + [self.b]


class Layer:
    """
    A layer is just multiple neurons stacked in parallel.
    nin = number of inputs
    nout = number of neurons in this layer
    """

    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        # return list of outputs (unless it's a 1-neuron layer)
        outs = [n(x) for n in self.neurons]
        return outs

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]


class MLP:
    """
    Multilayer Perceptron (tiny neural net).
    layer_sizes = e.g., [3, 4, 4, 1]
    means:
       input size = 3
       hidden1 = 4 neurons
       hidden2 = 4 neurons
       output = 1 neuron
    """

    def __init__(self, nin, layers):
        # layers is a list like [4,4,1]
        sz = [nin] + layers
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(layers))]

    def __call__(self, x):
        # pass forward through layers
        for layer in self.layers:
            x = layer(x)
        return x[0] if len(x) == 1 else x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
