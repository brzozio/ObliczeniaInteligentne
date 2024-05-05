import numpy as np

data = np.genfromtxt("mean_digit_convolution_train_data.txt", delimiter=";")
max = np.zeros(len(data))
for i in range(len(data)):
    max[i] = np.argmax(data[i])
labels = np.genfromtxt("train_labels.txt",delimiter=";")
acc = 0.0
for i in range(len(data)):
    if max[i] == labels[i]:
        acc += 1.0
print(acc/len(data))

data1 = np.genfromtxt("mean_digit_convolution_test_data.txt", delimiter=";")
max1 = np.zeros(len(data1))
for i in range(len(data1)):
    max1[i] = np.argmax(data1[i])
labels = np.genfromtxt("test_labels.txt",delimiter=";")
acc = 0.0
for i in range(len(data1)):
    if max1[i] == labels[i]:
        acc += 1.0
print(acc/len(data1))




