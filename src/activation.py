import numpy as np

## --------------------- Activation Functions --------------------- 
def linear(net, derivative=False):
    if derivative:
        return 1
    else:
        return net
    
def relu(net, derivative=False):
    if derivative:
        return 1 if net > 0 else 0
    else:
        return max(0, net)

def sigmoid(net, derivative=False):
    fsigmoid = (1 / (1 + np.exp(-net)))
    if derivative:
        return net * (1 - net)
    else:
        return fsigmoid
    
def softmax(net, numOfNet=1, derivative=False, target=False):
    if derivative:
        return net * (1 - net)
    else:
        sigma = 0
        for i in range (len(numOfNet)):
            sigma += np.exp(numOfNet[i])
        fsoftmax = np.exp(net) / sigma
        return fsoftmax