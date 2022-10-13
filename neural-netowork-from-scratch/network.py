# input layer
# inputs = [1.2, 5.1, 2.1]
# weight = [3.1, 2.1, 8.7]
# bias = 3
# output =0

inputs = [1, 2, 3]
weight = [0.2,0.8,-0.5]
bias = 2
output =0

for i,j in enumerate(inputs):
    output += j *weight[i]
print(output+bias)


# four inputs to 3 neurons
inputs = [1, 2, 3, 4]

# three neurons in the layer
# each neuron is associated with weight and bias


inputs = [1, 2, 3, 4]
weights = [[0.2,0.8,-0.5,1.0],
           [0.5,-0.91,0.26,-0.5],
           [-0.26,-0.27,0.17,0.87]]
biases = [2,3,0.5]
# input from four neurons to 3 neurons
# [inputs[0]*weight1+ inputs[1]*weight1+ inputs[2]*weight1+ inputs[3]*weight1 + bias1,
# inputs[0]*weight2+ inputs[1]*weight2+ inputs[2]*weight2+ inputs[3]*weight2 + bias2,
# inputs[0]*weight3+ inputs[1]*weight3+ inputs[2]*weight3+ inputs[3]*weight3 + bias3,
# ]

print(tuple(zip(weights, biases)))
layer_outputs = []
# for each neuron in thge current layer
for neuron_weights, neuron_bias in zip(weights, biases):
    # map each weight list with bias
    # [(weight,bias),....]
    # print(tuple(zip(weights, biases)))
    # reset the neuron output
    neuron_output =0
    for n_input, weight in zip(inputs, neuron_weights):
        # weight is then mapped with input
        # (input, weight)
        neuron_output += n_input * weight
        print(neuron_output)
    neuron_output+=bias
    # after weight x inpout add bias
    layer_outputs.append(neuron_output)


# dot product

import numpy as np

inputs = [1,2,3,2.5]
weights = [0.2,0.8,-0.5,1.0]
bias =2
output = np.dot(weights, inputs)+bias

inputs = [1,2,3,2.5]
weights = [[0.2,0.8,-0.5,1.0],
           [0.5,-0.91,0.26,-0.5],
           [-0.26,-0.27,0.17,0.87]]
bias =2
output = np.dot(inputs, weights )+bias


# batch processing

inputs = [[1,2,3,2.5],[2,5,-1,2],[-1.5,2.7,3.3,-0.8]]
weights = [[0.2,0.8,-0.5,1.0],
           [0.5,-0.91,0.26,-0.5],
           [-0.26,-0.27,0.17,0.87]]
bias =[2,3,0.5]
output = np.dot(inputs, np.array(weights).T )+bias

# multiple layers

inputs = [[1,2,3,2.5],[2,5,-1,2],[-1.5,2.7,3.3,-0.8]]
weights = [[0.2,0.8,-0.5,1.0],
           [0.5,-0.91,0.26,-0.5],
           [-0.26,-0.27,0.17,0.87]]
bias =[2,3,0.5]
weights2 = [[0.1,-0.14,0.5],
           [-0.5,0.12,-0.33],
           [-0.44,0.73,-0.13]]
bias2 =[-1, 2, -0.5 ]




layer1_output = np.dot(inputs, np.array(weights).T )+bias
layer2_output = np.dot(layer1_output, np.array(weights2).T )+bias2

# object oriented neural network
X = [[1,2,3,2.5],[2,5,-1,2],[-1.5,2.7,3.3,-0.8]]
# input featureset is denoted by capitle letter

# saving the model means saving the state of the weights and biases
# loading the model means restoring the weights and biases

import nnfs
nnfs.init()

class Layer_Dense:
    def __init__(self,n_inputs,n_neurons):
        self.weights = 0.10*np.random.randn(n_inputs,n_neurons)
        self.biases = np.zeros((1,n_neurons))
    def forward(self,inputs):
        self.output =  np.dot(inputs,self.weights) + self.biases


layer1 =  Layer_Dense(4,5)
layer2 =  Layer_Dense(5,2)

layer1.forward(X)
layer2.forward(layer1.output)
layer2.output


# activation functions

# step function -
# sigmiod - more granular output vanshing gradient problem
# rectified linear ReLU(x) more granular, fast

inputs = [0,2,-1, 3.3,-2.7, 1.1,2.2,-100]
output = []

# rectified linear actiation fucntion
for i in inputs:
    if i > 0:
        output.append(i)
    elif i < 0:
        output.append(0)

# rectifed linear object

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0,inputs)


def create_data(points, classes):
    X = np.zeros((points*classes,2))
    y = np.zeros(points*classes,dtype='uint8')
    for class_number in range(classes):
        ix = range(points*class_number, points*(class_number+1))
        r = np.linspace(0.0, 1, points)
        t = np.linspace(class_number*4, (class_number+1)*4,points) + np.random.randn(points)*0.2
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] =  class_number
    return X, y

import matplotlib.pyplot  as plt

X, y = create_data(100,3)
plt.scatter(X[:,0],X[:,1])

plt.scatter(X[:,0],X[:,1], c=y, cmap="brg")



# layer with activation function

layer1 = Layer_Dense(2, 5)
activation1 =  Activation_ReLU()
layer1.forward(X)
layer1.output
activation1.forward(layer1.output)
activation1.output