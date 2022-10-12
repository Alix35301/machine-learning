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

#layer
# neuron 1
bias1 = 0.5
weight1= 2
# neuron 2
bias2 = 0.6
weight2= 1
# neuron 3
bias3 =0.7
weight3 = 4

# input from four neurons to 3 neurons
[inputs[0]*weight1+ inputs[1]*weight1+ inputs[2]*weight1+ inputs[3]*weight1 + bias1,
inputs[0]*weight2+ inputs[1]*weight2+ inputs[2]*weight2+ inputs[3]*weight2 + bias2,
inputs[0]*weight3+ inputs[1]*weight3+ inputs[2]*weight3+ inputs[3]*weight3 + bias3,
]