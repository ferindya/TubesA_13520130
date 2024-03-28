import numpy as np

class Node:
    def __init__(self, bias, weight, activation_function, name):
        self.bias = bias
        self.weight = weight
        self.activation_function = activation_function
        self.name = name
        self.net = 0
        self.value = 0
        
    def calculate_net(self, input):
        self.net = np.dot(self.weight, input) + self.bias
        
    def activate_neuron(self, sum = None):
        if (self.name != "softmax"):
            self.value = self.activation_function(self.net)
        else:
            self.value = self.activation_function(self.net, sum)