from hidden_layer import HiddenLayer

class FFNN:
    '''Kelas untuk membuat model ffnn
	'''
    def __init__(self, case):
        self.input_size = case.get("model").get("input_size")
        self.nlayers = len(case.get("model").get("layers"))
        self.layers = HiddenLayer(case.get("model").get("layers"), case.get("weights"))

    def ffnn_model(case):
        '''Fungsi input untuk membuat model ffnn
	        input :  
			    case: data berupa model, input dan weight
    	   return:  
	    		valid_path
	    '''
        return 0
    
    def predict(self, input):
        output = []
        for i in range(len(input)):
            print("Input",i,":",input[i])
            output.append(self.layers.activate_hidden_layers(input[i]))
            print(self)
            print("Output:",output[-1].value)
            self.layers.restart_hidden_layer()
            print("-----------------------------------------------------------")
    
    def __str__(self):
        print(self.layers)
        return ''

if __name__ == "__main__":
    from input_json import file_name, open_json
    file_name = file_name(str(input("Masukin namfel:")))
    case, expect = open_json(file_name)
    model = FFNN(case)
    print(model)
    model.predict(case.get("input"))