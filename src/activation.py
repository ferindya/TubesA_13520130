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
        return fsigmoid * (1 - fsigmoid)
    else:
        return fsigmoid
    
def softmax(net, numOfNet, derivative=False):
    sigma = 0
    fsoftmax = np.exp(net) / sigma
    for i in range (len(numOfNet)):
        sigma += np.exp(numOfNet[i])
    if derivative:
        return fsoftmax * (1 - fsoftmax)
    else:
        return fsoftmax