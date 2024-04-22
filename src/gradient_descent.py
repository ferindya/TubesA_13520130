import numpy as np
from src.activation import linear, relu, sigmoid, softmax
from src.loss_function import sse, cross_entropy
# from activation import linear, relu, sigmoid, softmax
# from loss_function import sse, cross_entropy

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
                    #print(k,i,j,delta_w)
                    delta_w[k][i][j] += -learning_rate * activation_function(output[k][j], derivative=True) * input[k][i]
                else:
                    delta_w[k][i][j] += -learning_rate * sse(target[k][j], output[k][j], True) * activation_function(output[k][j], derivative=True) * input[k][i]

    return dssr_dw, delta_w

def gd_h(output, input, gd_h1,  weight, weight_next, learning_rate, activation):
    activation_function_list = {
            "linear" : linear,
            "relu" : relu,
            "sigmoid" : sigmoid,
            "softmax" : softmax,
            "None" : None
    }
    activation_function = activation_function_list[activation]

    de_dw = np.zeros_like(output)
    for i in range(len(output)):
        for j in range(len(output[i])):
            de_dw[i][j] += np.dot(gd_h1[i] * activation_function(output[i][j], derivative=True),weight_next[i][j])

    delta_w = np.zeros_like(weight)
    #if (activation != "softmax"):
    for k in range(len(input)):
        for i in range(len(input)): #input
            for j in range(len(input[i])):  #output
                delta_w[k][i][j] += learning_rate * np.dot(gd_h1[i] * activation_function(output[i][j], derivative=True), weight_next[i][j]) * input[k][i] 
    
    return de_dw, delta_w

if __name__ == "__main__":
    
    output_o = [
        #[0.573544416],[0.5609942702],[0.5610838429],
        [0.5476717749]]
    target_o = [
        #[1],[-1],[-1],
        [1]]
    weight_o = [
                # [[0.2184150084],	[0.2184150084]],
                # [[0.2184150084],	[0.2184150084]],
                # [[0.2184150084],	[0.2184150084]],
                [[0.2184150084],	[0.2184150084]]]
    input_o = [
            #    [0.677620292,	0.6790958157],
            #    [0.5604800631,	0.5621452944],
            #    [0.5621452944,	0.5621452944],
               [0.4378547056,	0.4378547056]]
    delta, result = gd_o(target_o, output_o, input_o, weight_o, 0.1, 'sigmoid')
    print(delta)
    #print(result)

    output_h = [
                #[0.6791786992,	0.6791786992],
                #[0.5621765009,	0.5621765009],
                #[0.5621765009,	0.5621765009],
                [0.4378234991,	0.4378234991]]
    weight_h = [
                # [[0.2464919039,	0.2464919039],
                #  [0.2498732144,	0.2498732144],
                #  [0.2498732144,	0.2498732144]],
                # [[0.2464919039,	0.2464919039],
                #  [0.2498732144,	0.2498732144],
                #  [0.2498732144,	0.2498732144]],
                # [[0.2464919039,	0.2464919039],
                #  [0.2498732144,	0.2498732144],
                #  [0.2498732144,	0.2498732144]],
                [[0.2464919039,	0.2464919039],
                 [0.2498732144,	0.2498732144],
                 [0.2498732144,	0.2498732144]]]
    input_h = [
            #    [1,	1,	1],
            #    [1,	1,	-1],
            #    [1,	-1,	1],
               [1,	-1, 1]]
                    #      1x5       1x3     1x2    1x1x3x4   1x1x5x2                  1x1x5
                    #      1x2       1x3     1x1    1x1x3x2   1x1x2x1                  1x1x2
    de_h, result_h = gd_h(input_o, input_h, delta, weight_h, weight_o, 0.1, 'sigmoid')
    print(result_h)
    print(de_h)
    #print(result_h)
    #print(weight_h)