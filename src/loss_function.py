import numpy as np

# For ReLU, sigmoid, and linear
def sse(target, output, derivative=False):
    if (derivative):
        return (output - target)
    else:
        return 0.5 * ((target - output) ** 2)
    
# For softmax
def cross_entropy(target, output, derivative=False):
    if (derivative):
        return -(target / output) + ((1 - target) / (1 - output))
    else:
        return -np.sum(target * np.log(output))