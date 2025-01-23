import random
from engine import Value
import numpy as np


class Neuron:

    def __init__(self, nin, activation="relu"):
        # nin : Number of inputs the neuron takes
        # each input has its own individual weights
        # single bias value
        self.b = Value(random.uniform(-1, 1), label="bias")
        self.activation = activation
        self.w = self.__initialize_weights(nin)
        
    
    def get_activation(self, value):
        """
        gets the activation function of choice
        """
        match self.activation:
            case "relu":
                return value.relu()
            case "tanh":
                return value.tanh()
            case "sigmoid":
                return value.sigmoid()
            case _:
                raise ValueError(f"Activation {self.activation} unknown")
    
    def __initialize_weights(self, nin):
        
        match self.activation:
            case "relu":
                w = np.random.normal(0, np.sqrt(2 / nin), nin)
                return [Value(i) for i in w]
            case "tanh" | "sigmoid":
                w = np.random.normal(-(1.0/np.sqrt(nin)), np.sqrt(1/nin), nin)
                return [Value(i) for i in w]
            case _:
                raise ValueError(f"Activation {self.activation} unknown")
    


    def __call__(self, x):
        # w *  x + b
        # the bias is only once 
        act = sum([inp*weight for inp, weight in zip(x, self.w)], self.b)
        out = self.get_activation(act)
        return out
    
    def parameters(self):
        return self.w  + [self.b]


class Layer:

    def __init__(self, nin, nout, activation="relu"):
        # nin: How much data is going into each neuron
        # nout : is bascally the number of neurons wanted 
        self.neurons = [Neuron(nin, activation) for _ in range(nout)]
    
    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs 
    
    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]


class MLP:
    """
    MLP takes in Layers so
    MLP(2, [(16, 'relu'), (16, 'relu'), (1, 'sigmoid')])
    """
    def __init__(self, nin, nout):
        # the output of the previous layer is the input for the next
        self.layers = []
        self.param_s = 0
        inp = nin
        for size, act in nout:
            self.layers.append(Layer(inp, size, act))
            inp = size
    

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    
    def zero_grad(self):
        for i in self.parameters():
            i.grad = 0
    
    def param_size(self):
        if self.param_s ==  0:
            self.param_s = len(self.parameters())
        return self.param_s