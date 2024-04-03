import os
import sys
from src.activation import linear, relu, sigmoid, softmax
from src.node import Node
import numpy as np

current_dir = os.path.dirname(__file__)
module_dir = os.path.join(current_dir, '..', 'function')
sys.path.append(module_dir)

class Layer:
    def __init__(self, weight, name, activation, n_neuron):
        activation_function = {
            "linear" : linear,
            "relu" : relu,
            "sigmoid" : sigmoid,
            "softmax" : softmax,
            "None" : None
        }
        self.bias = 1
        self.weight = np.transpose(weight)
        self.name = name
        self.n_neuron = int(n_neuron)
        self.activation_function = activation_function[activation]
        self.value = []
        self.net = []
        self.node = []
        self.generate()
    
    def generate(self):
        for i in range(self.n_neuron):
            self.node.append(Node(1, self.weight[i], self.activation_function, i+1))

    def activate_layer(self, input):
        if self.activation_function == softmax:
            for i in range(self.n_neuron):
                self.net.append(self.node[i].calculate_net(input))
            for i in range(self.n_neuron):
                self.node[i].update_net(self.net[i])
                self.value.append(self.node[i].activate_neuron(self.net))
        else:
            for i in range(self.n_neuron):
                self.net.append(self.node[i].calculate_net(input))
                self.value.append(self.node[i].activate_neuron())
    
    def restart_layer(self):
        self.value = []
        self.net = []
        for i in range(len(self.node)):
            self.node[i].restart_neuron()
    
    def __str__(self):
        print('Layer',self.name)
        print('')
        for i in range(self.n_neuron):
            print(self.node[i])
        return ''
    
if __name__ == "__main__":
    Layer1 = Layer([
                    [0.1, 0.2, 0.3],
                    [0.4, -0.5, 0.6],
                    [0.7, 0.8, -0.9]
                   ], 1, 'relu', 3)
    
    Layer1.activate_layer([-1.0, 0.5])
    print(Layer1)
    print(Layer1.value)
    