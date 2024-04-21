#from src.hidden_layer import HiddenLayer
from hidden_layer import HiddenLayer
import copy

class FFNN:
    '''Kelas untuk membuat model ffnn
	'''
    def __init__(self, case):
        self.input_size = case.get("model").get("input_size")
        self.nlayers = len(case.get("model").get("layers"))
        self.layers = HiddenLayer(case.get("model").get("layers"), case.get("weights"))
        self.noutput = self.layers.n_output
        self.output = []

    def ffnn_model(case):
        '''Fungsi input untuk membuat model ffnn
	        input :  
			    case: data berupa model, input dan weight
    	   return:  
	    		valid_path
	    '''
        return 0
    
    def model_activator(self, input):
        model_copy = copy.deepcopy(self)
        model_copy.layers.activate_hidden_layers(input)
        return model_copy
    
    # def forward_propagation_ffnn(self, input_file_name):
    #     checked_file_name = file_name(input_file_name)
    #     case, expect = open_json(checked_file_name)
    #     model = FFNN(case)
    #     input = case.get("input")
    #     output = model.predict(input)
    #     output_val = FFNN.output_value(self, output)
    #     sse = FFNN.f_sse(output_val, expect.get("output"))
    #     return model, input, output, sse
    
    def backpropgation(self, output):
        for i in range(self.nlayers):
            back_index = self.nlayers -1 -i
    
    def train():
        return 0

    def predict(self, input):
        output = [] 
        for i in range(len(input)):
            model_output = self.model_activator(input[i])
            output.append(model_output)
            self.layers.restart_hidden_layer()
        return output
    
    def output_value(self, model):
        output_val = []
        for i in range(len(model)):
            output_val.append(model[i].layers.layers[-1].value)
        self.output = output_val
        return output_val
    
    def f_sse(output, expect):
        sse = []
        for i in range(len(output)):
            sse.append(0)
            for j in range(len(output[i])):
                sse[i] += (output[i][j] - expect[i][j]) ** 2
        return sse

    def __str__(self):
        print(self.layers)
        return ''

if __name__ == "__main__":
    from input_json import file_name, open_json

    def forward_propagation_ffnn(input_file_name):
        checked_file_name = file_name(input_file_name)
        case, expect = open_json(checked_file_name)
        model = FFNN(case)
        input = case.get("input")
        output = model.predict(input)
        output_val = model.output_value(output)
        sse = FFNN.f_sse(output_val, expect.get("output"))
        return model, input, output, sse, output_val

    # file_name = file_name(str(input("Masukin namfel:")))
    # case, expect = open_json(file_name)
    # model = FFNN(case)
    #print(model)
    # output = model.predict(case.get("input"))
    model, input, output, sse, output_val = forward_propagation_ffnn('multilayer_softmax.json')
    for i in range(len(output)):
        print(output[i])
    # print(output)
    # print(expect)
    #print(FFNN.f_sse(output, expect.get("output")))