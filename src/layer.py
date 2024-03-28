import os
import sys
from activation import linear, relu, sigmoid, softmax
from node import Node

current_dir = os.path.dirname(__file__)
module_dir = os.path.join(current_dir, '..', 'function')
sys.path.append(module_dir)

class Layer:
    def __init__(self, bias, weight, name, neuron):
        activation_function = {
            "linear" : linear,
            "relu" : relu,
            "sigmoid" : sigmoid,
            "softmax" : softmax,
            "None" : None
        }
        self.bias = bias
        self.weight = weight
        self.name = name
        self.neuron = neuron
        self.activation_function = activation_function[name]
        self.value = []
        self.net = []
        self.node = []
        self.generate()
    
    def generate(self):
        for i in range(self.neuron):
            self.node.append(Node(self.bias[i], self.weight[i], self.activation_function, self.name))