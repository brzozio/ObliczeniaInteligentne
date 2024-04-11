import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.neural_network import MLPClassifier as MLP
from sklearn.svm import SVC
from voronoi import plot_decision_boundary, plot_decision_boundary_ax


hidden_neurons_MLP: np.array = [2,3,4,5,10,20,30,100]
c_parameter_SVM   : np.array = [1,3,10,20]

def granica_decyzyjna_SVM():
    for index in range(3):
        Data_train = np.genfromtxt(f"C:\\Users\\Michał\\Documents\\STUDIA\\II stopień, Informatyka Stosowana - inżynieria oprogramowania i uczenie maszynowe\\I sem\\Obliczenia inteligentne\\Projekt 1 - zadanie 2\\2_{index+1}.csv", delimiter=';')
        
        svm_fig, svm_ax = plt.subplots(4, 2, figsize=(10, 20))

        for i, c_param in enumerate(c_parameter_SVM):
            svc_linear_classifier = SVC(kernel='linear', C=c_param)
            svc_rbf_classifier = SVC(kernel='rbf', C=c_param)

            svc_linear_classifier.fit(Data_train[:, 0:2], Data_train[:, 2])
            svc_rbf_classifier.fit(Data_train[:, 0:2], Data_train[:, 2])

            # Granice decyzyjne dla SVM LINEAR
            plot_decision_boundary_ax(Data_train[:,0:2], axes_dec=svm_ax[i, 0], func=lambda X: svc_linear_classifier.predict(X))
            svm_ax[i, 0].set_title(f"Linear SVM (CSV: 2_{index+1}) (C={c_param})")

            # Granice decyzyjne dla SVM RBF
            plot_decision_boundary_ax(Data_train[:,0:2], axes_dec=svm_ax[ i, 1], func=lambda X: svc_rbf_classifier.predict(X))
            svm_ax[i, 1].set_title(f"RBF SVM (CSV: 2_{index+1})(C={c_param})")

        plt.subplots_adjust(hspace=0.6, wspace=0.5)
        plt.show()
def granica_decyzyjna_MLP():
    pass

if __name__ == "__main__":
    granica_decyzyjna_SVM()