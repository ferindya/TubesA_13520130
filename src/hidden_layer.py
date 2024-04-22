import numpy as np
from src.layer import Layer

class HiddenLayer:
    def __init__(self, layers, weights):
        self.n_layers = len(layers)
        self.n_output = layers[self.n_layers-1].get("number_of_neurons")

        self.layer_sizes=np.zeros(self.n_layers)
        # Update layer size
        for i in range(self.n_layers):
            self.layer_sizes[i] = layers[i].get("number_of_neurons")

        self.layers = []
        # Update layers
        for i in range(self.n_layers):
            new_layers = Layer(weights[i], i+1, layers[i].get("activation_function"), self.layer_sizes[i])
            self.layers.append(new_layers)
    
    def activate_hidden_layers(self, input):
        for i in range(self.n_layers):
            if (i == 0):
                self.layers[i].activate_layer(input)
            else:
                self.layers[i].activate_layer(self.layers[i-1].value)
        #return self.layers[-1]

    def restart_hidden_layer(self):
        for i in range(self.n_layers):
            self.layers[i].restart_layer()
        return 0
    
    def __str__(self):
        for i in range(self.n_layers):
            print(self.layers[i])
        return ''

if __name__ == "__main__":
    from input_json import file_name, open_json
    file_name = file_name(str(input("Masukin namfel:")))
    #file_name = '../models/relu.json'
    case, expect = open_json(file_name)
    hiddenLayer1 = HiddenLayer(case.get("model").get("layers"), case.get("weights"))
    print(hiddenLayer1)
    hiddenLayer1.activate_hidden_layers(case.get("input")[0])
    print(hiddenLayer1)