import numpy as np
from activation import linear, relu, sigmoid, softmax
from loss_function import sse, cross_entropy

def gd_o (target, output, input, weight, learning_rate, activation):

    activation_function_list = {
            "linear" : linear,
            "relu" : relu,
            "sigmoid" : sigmoid,
            "softmax" : softmax,
            "None" : None
    }
    activation_function = activation_function_list[activation]

    dssr_dw = np.zeros_like(output)
    for i in range(len(output)):
        for j in range(len(output[i])):
            if (activation == "softmax"):
                dssr_dw[i][j] += -1 * cross_entropy(target[i][j], output[i][j], True) * activation_function(output[i][j], derivative=True)
            else:
                dssr_dw[i][j] += -1* sse(target[i][j], output[i][j], True) * activation_function(output[i][j], derivative=True)
            
    delta_w = np.zeros_like(weight)
    for k in range(len(input)):
        for i in range(len(weight[k])): #input
            for j in range(len(weight[k][i])):  #output
                if (activation == "softmax"):
                    delta_w[k][i][j] += -learning_rate * cross_entropy(target[k][j], output[k][j], True) * activation_function(output[k][j], derivative=True) * input[k][i]
                else:
                    delta_w[k][i][j] += -learning_rate * sse(target[k][j], output[k][j], True) * activation_function(output[k][j], derivative=True) * input[k][i]

    return dssr_dw, delta_w

def gd_h (output, input, gd_h1, weight, activation):

    activation_function_list = {
            "linear" : linear,
            "relu" : relu,
            "sigmoid" : sigmoid,
            "softmax" : softmax,
            "None" : None
    }
    activation_function = activation_function_list[activation]

    de_dw = np.zeros_like(weight)
    if (activation != "softmax"):
        for k in range(len(input)):
            print('k:',k)
            for i in range(len(weight[k])): #input
                print('i:',i)
                for j in range(len(weight[k][i])):  #output
                    print('j:',j)
                    de_dw[k][i][j] += gd_h1[k] * activation_function(output[k][j], derivative=True) * input[k][i] * weight[k][i][j]
    
    return de_dw

if __name__ == "__main__":
    
    output_o = [[0.5840907676], [0.5698130028], [0.5698130028], [0.5545104221]]
    target_o = [[1],[-1],[-1],[1]]
    weight_o = [[[0.25],
               [0.25]],
              [[0.25], 
               [0.25]],
              [[0.25], 
               [0.25]],
              [[0.25], 
               [0.25]]]
    input_o = [[0.6791786992,	0.6791786992], [0.5621765009,	0.5621765009], [0.5621765009,	0.5621765009], [0.4378234991,	0.4378234991]]
    delta, result = gd_o(target_o, output_o, input_o, weight_o, 0.1, 'sigmoid')
    #print(delta)
    print(result)

    output_h = [[0.6791786992,	0.6791786992],
                [0.5621765009,	0.5621765009],
                [0.5621765009,	0.5621765009],
                [0.4378234991,	0.4378234991]]
    weight_h = [[[0.25,	0.25],	
                 [0.25,	0.25],	
                 [0.25,	0.25]],
                [[0.25,	0.25],	
                 [0.25,	0.25],	
                 [0.25,	0.25]],
                [[0.25,	0.25],	
                 [0.25,	0.25],	
                 [0.25,	0.25]],
                [[0.25,	0.25],	
                 [0.25,	0.25],	
                 [0.25,	0.25]]]
    input_h = [[1,	1,	1],
               [1,	1,	-1],
               [1,	-1,	1],
               [1,	-1, 1]]
    result_h = gd_h(output_h, input_h, delta, weight_h, 'sigmoid')
    print(result_h)
    #print(weight_h)