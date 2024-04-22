#from src.hidden_layer import HiddenLayer
#from src.gradient_descent import gd_h, gd_o
#from src.loss_function import sse, cross_entropy
from hidden_layer import HiddenLayer
from gradient_descent import gd_h, gd_o
from loss_function import sse, cross_entropy
from input_json import file_name, open_json, create_json, update_json
import copy
import numpy as np

class FFNN:
    '''Kelas untuk membuat model ffnn
	'''
    def __init__(self, case):
        self.input_size = case.get("model").get("input_size")
        self.nlayers = len(case.get("model").get("layers"))
        self.layers = HiddenLayer(case.get("model").get("layers"), case.get("weights"))
        self.noutput = self.layers.n_output
        self.weight = case.get("weights")
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
    
    def forward_propagation_ffnn(input_file_name):
        checked_file_name = file_name(input_file_name)
        case, expect = open_json(checked_file_name)
        model = FFNN(case)
        input = case.get("input")
        output = model.predict(input)
        output_val = model.output_value(output)
        #sse = FFNN.f_sse(output_val, expect.get("output"))
        return model, input, output, output_val, case, expect 
    
    def backpropgation(learning_rate, target, batch, weight):
        updated_weights = np.zeros_like(batch[0].weight)
        delta_all = []
        activation = batch[0].layers.layers[-1].activation_function_name
        
        for i in range(batch[0].nlayers):
            back_index = batch[0].nlayers -1 -i

            if i == 0 : # output layer
                output = []
                input = []
                weight_dim = []

                for i in range(len(batch)):
                    output.append(batch[i].layers.layers[back_index].value)
                    if (batch[0].nlayers > 1 and i != batch[0].nlayers-1):
                        input.append([1] + batch[i].layers.layers[back_index-1].value)
                    else:
                        input.append(batch[i].layers.layers[back_index-1].value)
                    weight_dim.append(weight[-1])
                delta, update = gd_o(target, output, input, weight_dim, learning_rate, activation)
                to_update_weights = update[0]
                for i in range(1, len(update)):
                    to_update_weights += update[i]
                updated_weights[back_index] = to_update_weights
                delta_all.append(delta)

            else: #hidden layer
                output = []
                input = []
                weight_dim = []
                weight_dim_next = []
                
                for j in range(len(batch)):
                    if (batch[0].nlayers > 1): #and i != batch[0].nlayers-1):
                        input.append([1] + batch[j].layers.layers[back_index-1].value)
                        output.append([1] + batch[j].layers.layers[back_index].value)
                    else:
                        input.append(batch[j].layers.layers[back_index-1].value)
                        output.append(batch[j].layers.layers[back_index].value)  # Next layer's output
                    weight_dim.append(weight[back_index])
                    weight_dim_next.append(weight[back_index + 1])

                # print('output:', output)
                # print('input:', input)
                # print('weight_dim:', weight_dim)
                # print('weight_dim_next:', weight_dim_next)
                # print('delta:', delta_all[-1])
            
                # Calculate delta and weight update for the hidden layer
                delta, update = gd_h(output, input, delta_all[-1], weight_dim, weight_dim_next, learning_rate, activation)
                to_update_weights = update[0]
                for i in range(1, len(update)):
                    to_update_weights += update[i]
                #print(type(to_update_weights))
                updated_weights[back_index] = to_update_weights
                delta_all.append(delta)
            
        return updated_weights

    def create_and_randomize_batch(total_input, batch_size):
        batches_num=[]
        total_input_array = np.arange(total_input)
        np.random.shuffle(total_input_array)

        total_of_batches=total_input // batch_size
        for i in range(total_of_batches):
            batch_indices = total_input_array[i * batch_size: (i + 1) * batch_size]
            batches_num.append(batch_indices)
        if total_input % batch_size != 0:
            batch_indices = total_input_array[total_of_batches * batch_size:]
            batches_num.append(batch_indices)

        return batches_num
    
    def train(input_file_name):
        # variables
        model, input, output, output_val, case, expect = FFNN.forward_propagation_ffnn(input_file_name)
        weight = case.get("weights")
        target = case.get("target")
        learning_parameters = case.get("learning_parameters")
        learning_rate = learning_parameters.get("learning_rate")
        batch_size = learning_parameters.get("batch_size")
        max_iteration = learning_parameters.get("max_iteration")
        error_threshold = learning_parameters.get("error_threshold")
        stopped_by = "max_iteration"
        error = error_threshold+1

        temporary_file_name = 'temporary.json'
        create_json(case, expect, temporary_file_name)

        for i in range(max_iteration):
            if i  != 0 :
            #     model, input, output, output_val, case = FFNN.forward_propagation_ffnn(input_file_name)
            #     weight = case.get("weights")
            # else :
                model, input, output, output_val, case, expect = FFNN.forward_propagation_ffnn(temporary_file_name)
                weight = case.get("weights")
            
            mse = 0
            if model.layers.layers[-1].activation_function_name == "softmax":
                for j in range(len(output_val)):
                    error = 0
                    for k in range(len(output_val[j])):
                        error += cross_entropy(case.get("target")[j][k], output_val[j][k])
                    mse += error
            else:
                for j in range(len(output_val)):
                    error = 0
                    for k in range(len(output_val[j])):
                        error += sse(case.get("target")[j][k], output_val[j][k])
                    mse += error
            mse /= output_val
            
            if (error <= error_threshold):
                stopped_by == "error_threshold"
                break
            else:
                # do backpropagation mini-batch
                batches = FFNN.create_and_randomize_batch(len(input), batch_size)
                weight_updates= np.zeros_like(weight)

                for batch in batches:
                    #proses backpropagation sampe update per batch
                    batch_weights_updates = np.zeros_like(weight)
                    batch_ffnn = []
                    for i in range(len(batch)):
                        batch_ffnn.append(output[i])
                    batch_weights_updates = FFNN.backpropgation(learning_rate, target, batch_ffnn, weight)

                    #save weight updates for each batch
                    #print(batch_weights_updates)
                    weight_updates += batch_weights_updates
                    
                
                #update weights
                #print(weight_updates)
                new_weight = weight + weight_updates

                #print(new_weight)

                #overwrite temporary files for next batch
                update_weight_final = []

                for i in range(len(new_weight)):
                    update_weight_final.append(new_weight[i].tolist())

                update_json(temporary_file_name, update_weight_final)

        case, expect = open_json(temporary_file_name)
        final_weights = case.get("weights")

        return stopped_by, final_weights

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

    # def forward_propagation_ffnn(input_file_name):
    #     checked_file_name = file_name(input_file_name)
    #     case, expect = open_json(checked_file_name)
    #     model = FFNN(case)
    #     input = case.get("input")
    #     output = model.predict(input)
    #     output_val = model.output_value(output)
    #     sse = FFNN.f_sse(output_val, expect.get("output"))
    #     return model, input, output, sse, output_val

    # file_name = file_name(str(input("Masukin namfel:")))
    # case, expect = open_json(file_name)
    # model = FFNN(case)
    # print(model)
    # output = model.predict(case.get("input"))
    model, input, output, output_val, case, expect = FFNN.forward_propagation_ffnn('multilayer_softmax.json')
    for i in range(len(output)):
        print(output[i])
    # print(output)
    # print(expect)
    #print(FFNN.f_sse(output, expect.get("output")))

    #stopped_by, final_weights = FFNN.train('b_softmax_two_layer.json')
    #stopped_by, final_weights = FFNN.train('b_relu.json') #aman
    #stopped_by, final_weights = FFNN.train('b_mlp.json')
    #stopped_by, final_weights = FFNN.train('b_sigmoid.json')
    stopped_by, final_weights = FFNN.train('b_softmax.json')
    #stopped_by, final_weights = FFNN.train('b_linear.json') #aman
    #stopped_by, final_weights = FFNN.train('b_linear_small_lr.json') #aman
    #stopped_by, final_weights = FFNN.train('b_linear_two_iteration.json') #aman
    print(stopped_by)
    print(final_weights)