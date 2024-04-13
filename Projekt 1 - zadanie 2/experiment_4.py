import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.neural_network import MLPClassifier as MLP
from sklearn.svm import SVC
from voronoi import plot_decision_boundary, plot_decision_boundary_ax
from sklearn.metrics import accuracy_score, confusion_matrix

hidden_neurons_MLP: np.array = [2,6,10,30]


if __name__ == "__main__":
    Data = np.genfromtxt(f"C:\\Users\\Michał\\Documents\\STUDIA\\II stopień, Informatyka Stosowana - inżynieria oprogramowania i uczenie maszynowe\\I sem\\Obliczenia inteligentne\\Projekt 1 - zadanie 2\\2_{index+2}.csv", delimiter=';')
    Data_test  = Data[260:300,:]
    Data_train_exp_3 = Data[0:60,:]
    Data_train_exp_2 = Data[0:260,:]