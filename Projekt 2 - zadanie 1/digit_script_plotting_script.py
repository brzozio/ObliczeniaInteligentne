import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import numpy as np

mnist = datasets.MNIST(
    root='data',
    train=True,
    download=True,
    transform= transforms.ToTensor()
)

domain_convolution = np.genfromtxt("mean_digit_convolution_train_data.txt", delimiter=';')
domain_differential = np.genfromtxt("raw_differential_train_data.txt", delimiter=';')

digit_id = 22

image = mnist[digit_id][0][0]
fig, ax = plt.subplots(2,2)
ax[0,0].imshow(image, cmap='gray')
ax[0,0].set_title("Wizualizacja graficzna")

ax[1,0].plot(range(28), domain_differential[digit_id][0:28], linestyle='-', marker=6)
ax[1,0].set_xticks(range(0,28,4))
ax[1,0].set_yticks(range(0,8,2))

ax[1,0].invert_yaxis()
ax[1,0].set_title("Ilość krawędzi pionowych")

ax[0,1].plot(domain_differential[digit_id][28:56],range(28,0,-1), linestyle='-', marker=4)
ax[0,1].set_yticks(range(4,32,4),range(24,-4,-4))
ax[0,1].set_xticks(range(0,8,2))
ax[0,1].set_title("Ilość krawędzi poziomych")

ax[1,1].bar(range(10), domain_convolution[digit_id], color='red')
ax[1,1].set_xticks(range(10))
ax[1,1].set_yticks(range(-3,4,1))
ax[1,1].set_title("Zbieżność z średnimi cyframi")

plt.show()