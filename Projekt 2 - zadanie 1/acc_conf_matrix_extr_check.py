import numpy as np

data = np.genfromtxt("mean_digit_convolution_test_data.txt", delimiter=";")
max = np.zeros(len(data))
for i in range(len(data)):
    max[i] = np.argmax(data[i])
print(max)
