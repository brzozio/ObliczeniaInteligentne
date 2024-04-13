import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.neural_network import MLPClassifier as MLP
from sklearn.svm import SVC
from voronoi import plot_decision_boundary, plot_decision_boundary_ax
from sklearn.metrics import accuracy_score


hidden_neurons_MLP: np.array = [2,6,10,30]
c_parameter_SVM   : np.array = [0.1,1.0,10.0,100.0,1000.0]

def granica_decyzyjna_SVM():
    svm_fig, svm_ax = plt.subplots(2, 3, figsize=(10, 20))
    for index in range(3):
        max_accuracy_lin          : float = 0.0
        max_accuracy_rbf          : float = 0.0
        c_param_best_accuracy_lin : float = 0.0
        c_param_best_accuracy_rbf : float = 0.0

        Data_train = np.genfromtxt(f"C:\\Users\\Michał\\Documents\\STUDIA\\II stopień, Informatyka Stosowana - inżynieria oprogramowania i uczenie maszynowe\\I sem\\Obliczenia inteligentne\\Projekt 1 - zadanie 2\\2_{index+1}.csv", delimiter=';')
        

        for i, c_param in enumerate(c_parameter_SVM):
            svc_linear_classifier = SVC(kernel='linear', C=c_param)
            svc_rbf_classifier = SVC(kernel='rbf', C=c_param)

            svc_linear_classifier.fit(Data_train[:, 0:2], Data_train[:, 2])
            svc_rbf_classifier.fit(Data_train[:, 0:2], Data_train[:, 2])

            #Wyliczanie 'accuracy' dla klasyfikatora rbf
            temp_labels_rbf = svc_rbf_classifier.predict(Data_train[:,0:2])
            accuracy_rbf = accuracy_score(temp_labels_rbf, Data_train[:,2])
            print(f"Accuracy RBF: {accuracy_rbf}")
            if accuracy_rbf > max_accuracy_rbf:
                max_accuracy_rbf          = accuracy_rbf
                c_param_best_accuracy_rbf = c_param

            #Wyliczanie 'accuracy' dla klasyfiaktora linear
            temp_labels_lin = svc_linear_classifier.predict(Data_train[:,0:2])
            accuracy_lin = accuracy_score(temp_labels_lin, Data_train[:,2])
            print(f"Accuracy LIN: {accuracy_lin}")
            if accuracy_lin > max_accuracy_lin:
                max_accuracy_lin          = accuracy_rbf
                c_param_best_accuracy_lin = c_param


        # Granice decyzyjne dla SVM LINEAR - best accuracy
        svc_linear_classifier = SVC(kernel='linear', C=c_param_best_accuracy_lin)
        svc_linear_classifier.fit(Data_train[:, 0:2], Data_train[:, 2])
        plot_decision_boundary_ax(Data_train[:,0:2], axes_dec=svm_ax[0,index], func=lambda X: svc_linear_classifier.predict(X))
        svm_ax[0,index].set_title(f"Linear SVM (CSV: 2_{index+1}) (C={c_param_best_accuracy_lin}) (Acc={max_accuracy_lin})")

        # Granice decyzyjne dla SVM RBF - best accuracy
        svc_rbf_classifier = SVC(kernel='rbf', C=c_param_best_accuracy_rbf)
        svc_rbf_classifier.fit(Data_train[:, 0:2], Data_train[:, 2])
        plot_decision_boundary_ax(Data_train[:,0:2], axes_dec=svm_ax[1,index], func=lambda X: svc_rbf_classifier.predict(X))
        svm_ax[1,index].set_title(f"RBF SVM (CSV: 2_{index+1})(C={c_param_best_accuracy_rbf}) (Acc={max_accuracy_rbf})")

    plt.subplots_adjust(hspace=0.6, wspace=0.5)
    plt.show()

def granica_decyzyjna_MLP():
    svm_fig, svm_ax = plt.subplots(4, 3, figsize=(10, 20))
    for index in range(3):

        max_accuracy_identity  : float = 0.0
        max_accuracy_logistic  : float = 0.0
        max_accuracy_tanh      : float = 0.0
        max_accuracy_relu      : float = 0.0
    
        hidden_n_layers_best_accuracy_identity : float = 0.0
        hidden_n_layers_best_accuracy_logistic : float = 0.0
        hidden_n_layers_best_accuracy_tanh     : float = 0.0
        hidden_n_layers_best_accuracy_relu     : float = 0.0


        Data_train = np.genfromtxt(f"C:\\Users\\Michał\\Documents\\STUDIA\\II stopień, Informatyka Stosowana - inżynieria oprogramowania i uczenie maszynowe\\I sem\\Obliczenia inteligentne\\Projekt 1 - zadanie 2\\2_{index+1}.csv", delimiter=';')

        for i, hidden_neurons_param in enumerate(hidden_neurons_MLP):
        #Wyliczanie wartości 'accuracy'
            mlp_classifier_identity = MLP(hidden_layer_sizes=hidden_neurons_param, max_iter=100000, n_iter_no_change=100000, tol=0, solver='sgd', activation='identity')   #zgodnie z wymaganiami projektowymi ustawione parametry
            mlp_classifier_identity.fit(Data_train[:, 0:2], Data_train[:, 2])
           
            mlp_classifier_logistic = MLP(hidden_layer_sizes=hidden_neurons_param, max_iter=100000, n_iter_no_change=100000, tol=0, solver='sgd', activation='logistic')   #zgodnie z wymaganiami projektowymi ustawione parametry
            mlp_classifier_logistic.fit(Data_train[:, 0:2], Data_train[:, 2])
            
            mlp_classifier_tanh = MLP(hidden_layer_sizes=hidden_neurons_param, max_iter=100000, n_iter_no_change=100000, tol=0, solver='sgd', activation='tanh')   #zgodnie z wymaganiami projektowymi ustawione parametry
            mlp_classifier_tanh.fit(Data_train[:, 0:2], Data_train[:, 2])
            
            mlp_classifier_relu = MLP(hidden_layer_sizes=hidden_neurons_param, max_iter=100000, n_iter_no_change=100000, tol=0, solver='sgd', activation='relu')   #zgodnie z wymaganiami projektowymi ustawione parametry
            mlp_classifier_relu.fit(Data_train[:, 0:2], Data_train[:, 2])

            temp_labels = mlp_classifier_identity.predict(Data_train[:,0:2])
            accuracy = accuracy_score(temp_labels, Data_train[:,2])
            if accuracy > max_accuracy_identity:
                max_accuracy_identity                  = accuracy
                hidden_n_layers_best_accuracy_identity = hidden_neurons_param
            
            temp_labels = mlp_classifier_logistic.predict(Data_train[:,0:2])
            accuracy = accuracy_score(temp_labels, Data_train[:,2])
            if accuracy > max_accuracy_logistic:
                max_accuracy_logistic                  = accuracy
                hidden_n_layers_best_accuracy_logistic = hidden_neurons_param
            
            temp_labels = mlp_classifier_tanh.predict(Data_train[:,0:2])
            accuracy = accuracy_score(temp_labels, Data_train[:,2])
            if accuracy > max_accuracy_tanh:
                max_accuracy_tanh                  = accuracy
                hidden_n_layers_best_accuracy_tanh = hidden_neurons_param
            
            temp_labels = mlp_classifier_relu.predict(Data_train[:,0:2])
            accuracy = accuracy_score(temp_labels, Data_train[:,2])
            if accuracy > max_accuracy_relu:
                max_accuracy_relu                  = accuracy
                hidden_n_layers_best_accuracy_relu = hidden_neurons_param
           


        #Plotowanie granicy decyzyjnej
        mlp_classifier_identity = MLP(hidden_layer_sizes=hidden_n_layers_best_accuracy_identity, max_iter=100000, n_iter_no_change=100000, tol=0, solver='sgd', activation='identity')   #zgodnie z wymaganiami projektowymi ustawione parametry
        mlp_classifier_identity.fit(Data_train[:, 0:2], Data_train[:, 2])
        plot_decision_boundary_ax(Data_train[:,0:2], axes_dec=svm_ax[0,index], func=lambda X: mlp_classifier_identity.predict(X))
        svm_ax[0,index].set_title(f"IDENTITY MLP (CSV: 2_{index+1}) (Neurons={hidden_n_layers_best_accuracy_identity}) (Acc={max_accuracy_identity})")

        mlp_classifier_logistic = MLP(hidden_layer_sizes=hidden_n_layers_best_accuracy_logistic, max_iter=100000, n_iter_no_change=100000, tol=0, solver='sgd', activation='logistic')   #zgodnie z wymaganiami projektowymi ustawione parametry
        mlp_classifier_logistic.fit(Data_train[:, 0:2], Data_train[:, 2])
        plot_decision_boundary_ax(Data_train[:,0:2], axes_dec=svm_ax[1,index], func=lambda X: mlp_classifier_logistic.predict(X))
        svm_ax[1,index].set_title(f"LOGISTIC MLP (CSV: 2_{index+1})(Neurons={hidden_n_layers_best_accuracy_logistic}) (Acc={max_accuracy_logistic}")
            

        mlp_classifier_tanh = MLP(hidden_layer_sizes=hidden_n_layers_best_accuracy_tanh, max_iter=100000, n_iter_no_change=100000, tol=0, solver='sgd', activation='tanh')   #zgodnie z wymaganiami projektowymi ustawione parametry
        mlp_classifier_tanh.fit(Data_train[:, 0:2], Data_train[:, 2])    
        plot_decision_boundary_ax(Data_train[:,0:2], axes_dec=svm_ax[2,index], func=lambda X: mlp_classifier_tanh.predict(X))
        svm_ax[2,index].set_title(f"TANH MLP (CSV: 2_{index+1})(Neurons={hidden_n_layers_best_accuracy_tanh}) (Acc={max_accuracy_tanh}")
            

        mlp_classifier_relu = MLP(hidden_layer_sizes=hidden_n_layers_best_accuracy_relu, max_iter=100000, n_iter_no_change=100000, tol=0, solver='sgd', activation='relu')   #zgodnie z wymaganiami projektowymi ustawione parametry
        mlp_classifier_relu.fit(Data_train[:, 0:2], Data_train[:, 2])
        plot_decision_boundary_ax(Data_train[:,0:2], axes_dec=svm_ax[3,index], func=lambda X: mlp_classifier_relu.predict(X))
        svm_ax[3,index].set_title(f"RELU MLP (CSV: 2_{index+1})(Neurons={hidden_n_layers_best_accuracy_relu}) (Acc={max_accuracy_relu}")

    plt.subplots_adjust(hspace=0.6, wspace=0.5)
    plt.show()

if __name__ == "__main__":
    #granica_decyzyjna_SVM()
    granica_decyzyjna_MLP()