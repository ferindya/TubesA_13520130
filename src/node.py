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
        net_array = np.concatenate(([1], input[0]),)
        self.net = np.dot(self.weight, net_array)
        return self.net
        
        
    def activate_neuron(self, sum = None):
        if (self.name != "softmax"):
            self.value = self.activation_function(self.net)
        else:
            self.value = self.activation_function(self.net, sum)
        return self.value
    
    def __repr__ (self):
        print('Node ', self.name)
        print('Weight: ',self.weight)
        #print(self.net)
        print('Value: ', self.value)
        return ''

if __name__ == "__main__":
    from activation import relu
    Node1 = Node(1, 
                [0.1, 0.2, 0.3]
                , relu, 2)
    Node1.calculate_net([[-1.0, 0.5]])
    Node1.activate_neuron()
    print(Node1)