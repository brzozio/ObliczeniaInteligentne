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
domain_tsne =  np.genfromtxt('TSNE_50_MNISTS_DATA.txt', delimiter=';')
domain_pca = np.genfromtxt('PCA_50_MNISTS_DATA.txt', delimiter=';')

digit_id = [1, 3, 5, 7, 20, 35, 18, 15, 31, 19]
origin = [0,0]

print(domain_tsne)

for d_id in digit_id:
    image = mnist[d_id][0][0]
    _, ax = plt.subplots(2,2, figsize=(10,20))
    ax[0,0].imshow(image, cmap='gray')
    ax[0,0].set_title(f'Wizualizacja graficzna')

    ax[1,0].plot(range(28), domain_differential[d_id][0:28], linestyle='-', marker=6)
    ax[1,0].set_xticks(range(0,28,4))
    ax[1,0].set_yticks(range(0,8,2))

    ax[1,0].invert_yaxis()
    ax[1,0].set_title("Ilość krawędzi pionowych")

    ax[0,1].plot(domain_differential[d_id][28:56], range(28, 0, -1), linestyle='-', marker=4)
    ax[0,1].set_yticks(range(4,32,4),range(24,-4,-4))
    ax[0,1].set_xticks(range(0,8,2))
    ax[0,1].set_title("Ilość krawędzi poziomych")

    ax[1,1].quiver(0,0, domain_tsne[d_id][0], domain_tsne[d_id][1], scale=1, scale_units='x')
    ax[1,1].set(xlim=(-2.2, 2.2), ylim=(-2.2, 2.2))
    ax[1,1].set_title("Wartości cech TSNE")

    plt.show()

    image = mnist[d_id][0][0]
    _, ax = plt.subplots(1,3, figsize=(18,5))

    ax[0].quiver(0,0, domain_pca[d_id][0], domain_pca[d_id][1], scale=1, scale_units='x')
    ax[0].set(xlim=(-2.2, 2.2), ylim=(-2.2, 2.2))
    ax[0].set_title("Wartości cech PCA")

    ax[1].imshow(image, cmap='gray')
    ax[1].set_title(f'Wizualizacja graficzna')

    ax[2].bar(range(10), domain_convolution[d_id], color='red')
    ax[2].set_xticks(range(10))
    ax[2].set_yticks(range(-3,4,1))
    ax[2].set_title("Zbieżność z średnimi cyframi")

    plt.show()