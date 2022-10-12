# input layer
inputs = [1.2, 5.1, 2.1]
weight = [3.1, 2.1, 8.7]
bias = 3
output =0

for i,j in enumerate(inputs):
    output += j *weight[i]
print(output+bias)