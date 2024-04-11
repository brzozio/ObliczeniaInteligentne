import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.neural_network import MLPClassifier as MLP
from sklearn.svm import SVC
from voronoi import plot_decision_boundary

#Data_train: np.ndarray = np.zeros((3,300,3))
#Data_train[0] = np.ndarray = np.genfromtxt("C:\\Users\\Michał\\Documents\\STUDIA\II stopień, Informatyka Stosowana - inżynieria oprogramowania i uczenie maszynowe\\I sem\\Obliczenia inteligentne\\Projekt 1 - zadanie 2\\2_1.csv", delimiter=';')
#Data_train[1] = np.ndarray = np.genfromtxt("C:\\Users\\Michał\\Documents\\STUDIA\II stopień, Informatyka Stosowana - inżynieria oprogramowania i uczenie maszynowe\\I sem\\Obliczenia inteligentne\\Projekt 1 - zadanie 2\\2_2.csv", delimiter=';')
#Data_train[2] = np.ndarray = np.genfromtxt("C:\\Users\\Michał\\Documents\\STUDIA\II stopień, Informatyka Stosowana - inżynieria oprogramowania i uczenie maszynowe\\I sem\\Obliczenia inteligentne\\Projekt 1 - zadanie 2\\2_3.csv", delimiter=';')

hidden_neurons_MLP: np.array = [2,3,4,5,10,20,30,100]
c_parameter_SVM   : np.array = [1,1.5,3,4.5,6,7.5,9,12]


for index in range(3):
    Data_train : np.array = np.genfromtxt(f"C:\\Users\\Michał\\Documents\\STUDIA\II stopień, Informatyka Stosowana - inżynieria oprogramowania i uczenie maszynowe\\I sem\\Obliczenia inteligentne\\Projekt 1 - zadanie 2\\2_{index+1}.csv", delimiter=';')
    #Porównać zmianę parametru 'activation' klasyfikatora opartego o sieć MLP
    mlp_classifier = MLP(hidden_layer_sizes=2, max_iter=100000, n_iter_no_change=100000, tol=0, solver='sgd')   #zgodnie z wymaganiami projektowymi ustawione parametry


    #Porównać zmianę parametru 'kernel' (linear oraz rbf) na klasyfikatorze SVM
    svc_linear_classifier = SVC(kernel='linear',C=3)
    svc_rbf_classifier    = SVC(kernel='rbf')

    svc_linear_classifier.fit(Data_train[:,0:2],Data_train[:,2])
    print(f'CLASSES ARE: {svc_linear_classifier.classes_}')
    plot_decision_boundary(Data_train[:,0:2], Data_train[:,2],  func=lambda X: svc_linear_classifier.predict(X))