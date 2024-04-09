import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN

Data: np.ndarray = np.zeros((6,300,3))
X: np.ndarray = np.genfromtxt("C:\\Users\\Michał\\Documents\\STUDIA\II stopień, Informatyka Stosowana - inżynieria oprogramowania i uczenie maszynowe\\I sem\\Obliczenia inteligentne\\Projekt 1 - zadanie 1\\1_1.csv", delimiter=';')
Data[0] = X
X: np.ndarray = np.genfromtxt("C:\\Users\\Michał\\Documents\\STUDIA\II stopień, Informatyka Stosowana - inżynieria oprogramowania i uczenie maszynowe\\I sem\\Obliczenia inteligentne\\Projekt 1 - zadanie 1\\1_2.csv", delimiter=';')
Data[1] = X
X: np.ndarray = np.genfromtxt("C:\\Users\\Michał\\Documents\\STUDIA\II stopień, Informatyka Stosowana - inżynieria oprogramowania i uczenie maszynowe\\I sem\\Obliczenia inteligentne\\Projekt 1 - zadanie 1\\1_3.csv", delimiter=';')
Data[2] = X
X: np.ndarray = np.genfromtxt("C:\\Users\\Michał\\Documents\\STUDIA\II stopień, Informatyka Stosowana - inżynieria oprogramowania i uczenie maszynowe\\I sem\\Obliczenia inteligentne\\Projekt 1 - zadanie 1\\2_1.csv", delimiter=';')
Data[3] = X
X: np.ndarray = np.genfromtxt("C:\\Users\\Michał\\Documents\\STUDIA\II stopień, Informatyka Stosowana - inżynieria oprogramowania i uczenie maszynowe\\I sem\\Obliczenia inteligentne\\Projekt 1 - zadanie 1\\2_2.csv", delimiter=';')
Data[4] = X
X: np.ndarray = np.genfromtxt("C:\\Users\\Michał\\Documents\\STUDIA\II stopień, Informatyka Stosowana - inżynieria oprogramowania i uczenie maszynowe\\I sem\\Obliczenia inteligentne\\Projekt 1 - zadanie 1\\2_3.csv", delimiter=';')
Data[5] = X

fig, axs = plt.subplots(3, 4)
Y: np.ndarray = np.zeros(300)
"""
best = [2,2,3,2,4,4]
worst = [4,4,6,9,3,9]
for i in range(6):

    Y = KMeans(n_clusters=best[i], n_init=10).fit_predict(Data[i, :, 0:2])
    axs[i%3,int(i/3)*2].scatter(Data[i, :, 0], Data[i, :, 1], s=6, c=Y)
    axs[i%3,int(i/3)*2].set_title("best from set "+str(int(i/3)+1)+"_"+str(i%3+1))

    Y = KMeans(n_clusters=worst[i], n_init=10).fit_predict(Data[i, :, 0:2])
    axs[i%3,int(i/3)*2+1].scatter(Data[i, :, 0], Data[i, :, 1], s=6, c=Y)
    axs[i%3,int(i/3)*2+1].set_title("worst from set "+str(int(i/3)+1)+"_"+str(i%3+1))

plt.show()
"""

best = [2,3,3,2,2,14]
worst = [9,2,2,21,21,2]
temp: DBSCAN

for i in range(6):

    temp = DBSCAN(eps=best[i], min_samples=10)
    temp.fit(Data[i, :, 0:2])
    Y = temp.labels_
    axs[i%3,int(i/3)*2].scatter(Data[i, :, 0], Data[i, :, 1], s=6, c=Y)
    axs[i%3,int(i/3)*2].set_title("best from set "+str(int(i/3)+1)+"_"+str(i%3+1))

    temp = DBSCAN(eps=worst[i], min_samples=10)
    temp.fit(Data[i, :, 0:2])
    Y = temp.labels_
    axs[i%3,int(i/3)*2+1].scatter(Data[i, :, 0], Data[i, :, 1], s=6, c=Y)
    axs[i%3,int(i/3)*2+1].set_title("worst from set "+str(int(i/3)+1)+"_"+str(i%3+1))

plt.show()