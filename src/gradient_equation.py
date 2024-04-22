from loss_function import sse, cross_entropy
from layer import Layer

def __init__(self, learning_rate, error_threshold, max_iter, batch_size, n_layer = None, n_neuron = None, activation_function = None):
    self.learning_rate = learning_rate
    self.error_threshold = error_threshold
    self.max_iter = max_iter
    self.batch_size = batch_size
    self.num_hidden_layer = 0
    self.hidden_layer = []
    self.input_size = 0
    self.input_layer = []
    self.output_layer = []

def update_delta(self, idx):
    # For hidden layer
    for i in range(self.num_hidden_layer):
        for node in self.hidden_layer[i].nodes:
            node.update_delta_bias(self.learning_rate)
            if (i == 0):
                node.update_delta_weight(True, self.input_layer[idx], None, self.learning_rate)
            else:
                node.update_delta_weight(False, None, self.hidden_layer[i-1].activation_function_value, self.learning_rate)
                
    # For output layer
    for node in self.output_layer.nodes:
        node.update_delta_bias(self.learning_rate)
        if (self.num_hidden_layer == 0):
            node.update_delta_weight(True, self.input_layer[idx], None, self.learning_rate)
        else:
            node.update_delta_weight(False, None, self.hidden_layer[self.num_hidden_layer-1].activation_function_value, self.learning_rate)
            
def update_bias_weight(self):
    # For hidden layer
    for idx in range(self.num_hidden_layer):
        for node in self.hidden_layer[idx].nodes:
            node.update_bias()
            node.update_weight()
        self.hidden_layer[idx].update_layer()
            
    # For output layer
    for node in self.output_layer.nodes:
        node.update_bias()
        node.update_weight()
    self.output_layer.update_layer()
    
