import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

Data: np.ndarray = np.zeros((6,300,3))
Data[0] = np.genfromtxt('1_1.csv', delimiter=';')
Data[1] = np.genfromtxt('1_2.csv', delimiter=';')
Data[2] = np.genfromtxt('1_3.csv', delimiter=';')
Data[3] = np.genfromtxt('2_1.csv', delimiter=';')
Data[4] = np.genfromtxt('2_2.csv', delimiter=';')
Data[5] = np.genfromtxt('2_3.csv', delimiter=';')

fig, axs = plt.subplots(3, 4)
best = [2,2,3,2,4,4]
worst = [4,4,6,9,3,9]

Y: np.ndarray = np.zeros((6,300))
for i in range(6):

    Y[i] = KMeans(n_clusters=best[i], n_init=10).fit_predict(Data[i, :, 0:2])
    axs[i%3,int(i/3)*2].scatter(Data[i, :, 0], Data[i, :, 1], s=6, c=Y[i])
    axs[i%3,int(i/3)*2].set_title("best from set "+str(int(i/3)+1)+"_"+str(i%3+1))

    Y[i] = KMeans(n_clusters=worst[i], n_init=10).fit_predict(Data[i, :, 0:2])
    axs[i%3,int(i/3)*2+1].scatter(Data[i, :, 0], Data[i, :, 1], s=6, c=Y[i])
    axs[i%3,int(i/3)*2+1].set_title("worst from set "+str(int(i/3)+1)+"_"+str(i%3+1))

plt.show()
