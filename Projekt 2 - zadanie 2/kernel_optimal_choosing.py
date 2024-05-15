import matplotlib.pyplot as plt
import numpy as np
from joblib import load

def plot_chart():
    kernels         = [3,5,7]
    mnist_channels  = [4,8,12]
    cifar_channels  = [5,10,15]

    mnist_obj = load("mnist_kernel_channels.joblib")
    cifar_obj = load("cifar_kernel_channels.joblib")
    #print(mnist_obj[0][1])

    # 3 ax po rozmiar kernela, dla kazdego kernela rozne rozmiary kanalow - modulo 3
    #MNIST and MNIST REDUCED
    _, ax_mnist = plt.subplots(2,3, figsize=(12,10))
    for i_kernel in range(3):
        for i_channel in range(3):
            ax_mnist[0, i_kernel].plot(range(len(mnist_obj[0][3*i_kernel+i_channel])), mnist_obj[0][3*i_kernel+i_channel], label=mnist_channels[i_channel])

            ax_mnist[1, i_kernel].plot(range(len(mnist_obj[1][3*i_kernel+i_channel])), mnist_obj[1][3*i_kernel+i_channel], label=mnist_channels[i_channel]) #reduced

        ax_mnist[0, i_kernel].set_title(f"Mnist - Kernel ({kernels[i_kernel]})")
        ax_mnist[1, i_kernel].set_title(f"Mnist Reduced - Kernel ({kernels[i_kernel]})")
        ax_mnist[0, i_kernel].legend()
        ax_mnist[1, i_kernel].legend()

    plt.show()
    
    #CIFAR and CIFAR REDUCED
    _, ax_cifar = plt.subplots(2,3, figsize=(12,10))
    for i_kernel in range(3):
        for i_channel in range(3):
            ax_cifar[0, i_kernel].plot(range(len(cifar_obj[0][3*i_kernel+i_channel])), cifar_obj[0][3*i_kernel+i_channel], label=cifar_channels[i_channel])

            ax_cifar[1, i_kernel].plot(range(len(cifar_obj[1][3*i_kernel+i_channel])), cifar_obj[1][3*i_kernel+i_channel], label=cifar_channels[i_channel]) #reduced

        ax_cifar[0, i_kernel].set_title(f"Cifar - Kernel ({kernels[i_kernel]})")
        ax_cifar[1, i_kernel].set_title(f"Cifar Reduced - Kernel ({kernels[i_kernel]})")
        ax_cifar[0, i_kernel].legend()
        ax_cifar[1, i_kernel].legend()
    
    plt.show()  

 


if __name__ == "__main__":
    plot_chart()