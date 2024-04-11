import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.neural_network import MLPClassifier as MLP
from sklearn.svm import SVC
from voronoi import plot_decision_boundary, plot_decision_boundary_ax
from sklearn.metrics import accuracy_score, confusion_matrix


knn_n_neighbours: np.array = [1,2,3,4,5,6,7,8,9,10,11,12,13,14]
c_parameter_SVM   : np.array = [0.001,0.01,0.1,1,10,100,1000,10000]
hidden_neurons_MLP: np.array = [2,6,10,30]


def KNN_granica_decyzyjna_accuracy():
    for index in range(2):
        Data = np.genfromtxt(f"C:\\Users\\Michał\\Documents\\STUDIA\\II stopień, Informatyka Stosowana - inżynieria oprogramowania i uczenie maszynowe\\I sem\\Obliczenia inteligentne\\Projekt 1 - zadanie 2\\2_{index+2}.csv", delimiter=';')
        Data_train = Data[0:60,:]
        Data_test  = Data[260:300,:]
        
        #Zbior testowy
        best_accuracy_test       : float = 0.0
        best_acc_n_neighb_test   : int   = 0
        worst_accuracy_test      : float = 1.0
        worst_acc_n_neighb_test  : int   = 1
        max_accuracy_test        : float = 0.0

        accuracy_plot_acc_test          = []
        accuracy_plot_n_neighbour_test  = []
        
        #Zbior treningowy
        worst_accuracy_train      : float = 1.0
        worst_acc_n_neighb_train  : int   = 1
        max_accuracy_train        : float = 0.0

        accuracy_plot_acc_train          = []
        accuracy_plot_n_neighbour_train  = []


        for i, n_neighbours_param in enumerate(knn_n_neighbours):
            knn_classifier = knn(n_neighbors=n_neighbours_param)

            knn_classifier.fit(Data_train[:, 0:2], Data_train[:, 2])

        #Wyliczanie accuracy
            #Test
            temp_labels_test = knn_classifier.predict(Data_test[:,0:2])
            accuracy_test = accuracy_score(temp_labels_test, Data_test[:,2])

            accuracy_plot_acc_test.append(accuracy_test)
            accuracy_plot_n_neighbour_test.append(n_neighbours_param)
            
            #Train
            temp_labels_train = knn_classifier.predict(Data_train[:,0:2])
            accuracy_train = accuracy_score(temp_labels_train, Data_train[:,2])

            accuracy_plot_acc_train.append(accuracy_train)
            accuracy_plot_n_neighbour_train.append(n_neighbours_param)

            #Wyznaczanie najlepszego i najgorszego accuracy
            if accuracy_test > best_accuracy_test:
                best_accuracy_test = accuracy_test
                best_acc_n_neighb_test = n_neighbours_param
               
            if accuracy_train < worst_accuracy_train:
                worst_accuracy_train = accuracy_test
                worst_acc_n_neighb_train = n_neighbours_param
            
            if accuracy_test < worst_accuracy_test:
                worst_accuracy_test = accuracy_test
                worst_acc_n_neighb_test = n_neighbours_param
            
        knn_fig, knn_ax = plt.subplots(3, 2, figsize=(10, 20)) #Train 0, Test 1
        acc_fig, acc_ax = plt.subplots()

#Granice decyzyjne
        
        max_accuracy_test  = max(accuracy_plot_acc_test)
        max_acc_n_neighb_test = knn_n_neighbours[accuracy_plot_acc_test.index(max_accuracy_test)]
        max_accuracy_train = max(accuracy_plot_acc_train)
        max_acc_n_neighb_train = knn_n_neighbours[accuracy_plot_acc_train.index(max_accuracy_train)]

        #Train
        knn_classifier_granica_TRAIN_BEST  = knn(n_neighbors=best_acc_n_neighb_test) #zgodnie z wymaganiami najlepszy n-neighbour na podstawie acc ze zbioru testowego
        knn_classifier_granica_TRAIN_BEST.fit(Data_train[:, 0:2], Data_train[:, 2])
        plot_decision_boundary_ax(Data_test[:,0:2], axes_dec=knn_ax[0, 0], func=lambda X: knn_classifier_granica_TRAIN_BEST.predict(X))
        knn_ax[0, 0].set_title(f"KNN train-set BEST (CSV: 2_{index+2}) (Neighbours={best_acc_n_neighb_test})")
        print(f'KNN train-set BEST (CSV: 2_{index+2}) Confusion matrix: {confusion_matrix(Data_train[:, 2],knn_classifier_granica_TRAIN_BEST.predict(Data_train[:, 0:2]))}')
        
        knn_classifier_granica_TRAIN_WORST = knn(n_neighbors=worst_acc_n_neighb_train)
        knn_classifier_granica_TRAIN_WORST.fit(Data_train[:, 0:2], Data_train[:, 2])
        plot_decision_boundary_ax(Data_train[:,0:2], axes_dec=knn_ax[1, 0], func=lambda X: knn_classifier_granica_TRAIN_WORST.predict(X))
        knn_ax[1, 0].set_title(f"KNN train-set WORST (CSV: 2_{index+2}) (Neighbours={worst_acc_n_neighb_train})")
        print(f'KNN train-set WORST (CSV: 2_{index+2}) Confusion matrix: {confusion_matrix(Data_train[:, 2],knn_classifier_granica_TRAIN_WORST.predict(Data_train[:, 0:2]))}')
        
        knn_classifier_granica_TRAIN_MAX = knn(n_neighbors=max_acc_n_neighb_train)
        knn_classifier_granica_TRAIN_MAX.fit(Data_train[:, 0:2], Data_train[:, 2])
        plot_decision_boundary_ax(Data_train[:,0:2], axes_dec=knn_ax[2, 0], func=lambda X: knn_classifier_granica_TRAIN_MAX.predict(X))
        knn_ax[2, 0].set_title(f"KNN train-set MAX (CSV: 2_{index+2}) (Neighbours={max_acc_n_neighb_train})")
        print(f'KNN train-set MAX (CSV: 2_{index+2}) Confusion matrix: {confusion_matrix(Data_train[:, 2],knn_classifier_granica_TRAIN_MAX.predict(Data_train[:, 0:2]))}')
         
        #Test
        knn_classifier_granica_TEST_BEST  = knn(n_neighbors=best_acc_n_neighb_test) #zgodnie z wymaganiami najlepszy n-neighbour na podstawie acc ze zbioru testowego
        knn_classifier_granica_TEST_BEST.fit(Data_train[:, 0:2], Data_train[:, 2])
        plot_decision_boundary_ax(Data_test[:,0:2], axes_dec=knn_ax[0, 1], func=lambda X: knn_classifier_granica_TEST_BEST.predict(X))
        knn_ax[0, 1].set_title(f"KNN test-set BEST (CSV: 2_{index+2}) (Neighbours={best_acc_n_neighb_test})")
        print(f'KNN test-set BEST (CSV: 2_{index+2}) Confusion matrix: {confusion_matrix(Data_train[:, 2],knn_classifier_granica_TEST_BEST.predict(Data_train[:, 0:2]))}')
         
        knn_classifier_granica_TEST_WORST = knn(n_neighbors=worst_acc_n_neighb_test)
        knn_classifier_granica_TEST_WORST.fit(Data_train[:, 0:2], Data_train[:, 2])
        plot_decision_boundary_ax(Data_train[:,0:2], axes_dec=knn_ax[1, 1], func=lambda X: knn_classifier_granica_TEST_WORST.predict(X))
        knn_ax[1, 1].set_title(f"KNN test-set WORST (CSV: 2_{index+2}) (Neighbours={worst_acc_n_neighb_test})")
        print(f'KNN test-set WORST (CSV: 2_{index+2}) Confusion matrix: {confusion_matrix(Data_train[:, 2],knn_classifier_granica_TEST_WORST.predict(Data_train[:, 0:2]))}')
        
        knn_classifier_granica_TEST_MAX = knn(n_neighbors=max_acc_n_neighb_test)
        knn_classifier_granica_TEST_MAX.fit(Data_train[:, 0:2], Data_train[:, 2])
        plot_decision_boundary_ax(Data_train[:,0:2], axes_dec=knn_ax[2, 1], func=lambda X: knn_classifier_granica_TRAIN_MAX.predict(X))
        knn_ax[2, 1].set_title(f"KNN test-set MAX (CSV: 2_{index+2}) (Neighbours={max_acc_n_neighb_test})")
        print(f'KNN test-set MAX (CSV: 2_{index+2}) Confusion matrix: {confusion_matrix(Data_train[:, 2],knn_classifier_granica_TRAIN_MAX.predict(Data_train[:, 0:2]))}')
         
        acc_ax.plot(accuracy_plot_n_neighbour_train, accuracy_plot_acc_train, 'o', color='green', linestyle='solid', linewidth=2, label="Train Data")
        acc_ax.plot(accuracy_plot_n_neighbour_test, accuracy_plot_acc_test,   'o', color='red',   linestyle='solid', linewidth=2, label="Test Data")
        acc_ax.set_xlabel("n_neighbours")
        acc_ax.legend()

        plt.subplots_adjust(hspace=0.6, wspace=0.5)
        plt.show()

def SVM_granica_decyzyjna_accuracy():
    """
    Wartości parametru C powinny się zmieniać wykładniczo, a na wykresie dobrze jest zastosować skalę logarytmiczną
    """
    for index in range(2):
        Data = np.genfromtxt(f"C:\\Users\\Michał\\Documents\\STUDIA\\II stopień, Informatyka Stosowana - inżynieria oprogramowania i uczenie maszynowe\\I sem\\Obliczenia inteligentne\\Projekt 1 - zadanie 2\\2_{index+2}.csv", delimiter=';')
        Data_train = Data[0:60,:]
        Data_test  = Data[260:300,:]
        
        #Zbior testowy
        best_accuracy_test       : float = 0.0
        best_acc_c_test          : int   = 0
        worst_accuracy_test      : float = 1.0
        worst_acc_c_test         : int   = 1
        max_accuracy_test        : float = 0.0

        accuracy_plot_acc_test = []
        accuracy_plot_c_test   = []
        
        #Zbior treningowy
        worst_accuracy_train      : float = 1.0
        worst_acc_c_train         : int   = 1
        max_accuracy_train        : float = 0.0

        accuracy_plot_acc_train  = []
        accuracy_plot_c_train    = []


        for i, c_param in enumerate(c_parameter_SVM):
            svc_linear_classifier = SVC(kernel='linear', C=c_param)

            svc_linear_classifier.fit(Data_train[:, 0:2], Data_train[:, 2])

        #Wyliczanie accuracy
            #Test
            temp_labels_test = svc_linear_classifier.predict(Data_test[:,0:2])
            accuracy_test = accuracy_score(temp_labels_test, Data_test[:,2])

            accuracy_plot_acc_test.append(accuracy_test)
            accuracy_plot_c_test.append(c_param)
            
            #Train
            temp_labels_train = svc_linear_classifier.predict(Data_train[:,0:2])
            accuracy_train = accuracy_score(temp_labels_train, Data_train[:,2])

            accuracy_plot_acc_train.append(accuracy_train)
            accuracy_plot_c_train.append(c_param)

            #Wyznaczanie najlepszego i najgorszego accuracy
            if accuracy_test > best_accuracy_test:
                best_accuracy_test = accuracy_test
                best_acc_c_test = c_param
               
            if accuracy_train < worst_accuracy_train:
                worst_accuracy_train = accuracy_test
                worst_acc_c_train = c_param
            
            if accuracy_test < worst_accuracy_test:
                worst_accuracy_test = accuracy_test
                worst_acc_c_test = c_param
            
        svm_fig, svm_ax = plt.subplots(3, 2, figsize=(10, 20)) #Train 0, Test 1
        acc_fig, acc_ax = plt.subplots()

#Granice decyzyjne
        
        max_accuracy_test  = max(accuracy_plot_acc_test)
        max_acc_c_test = c_parameter_SVM[accuracy_plot_acc_test.index(max_accuracy_test)]
        
        max_accuracy_train = max(accuracy_plot_acc_train)
        max_acc_c_train = c_parameter_SVM[accuracy_plot_acc_train.index(max_accuracy_train)]

        #Train
        svc_classifier_granica_TRAIN_BEST = SVC(kernel='linear', C=best_acc_c_test)
        svc_classifier_granica_TRAIN_BEST.fit(Data_train[:, 0:2], Data_train[:, 2])
        plot_decision_boundary_ax(Data_test[:,0:2], axes_dec=svm_ax[0, 0], func=lambda X: svc_classifier_granica_TRAIN_BEST.predict(X))
        svm_ax[0, 0].set_title(f"SVM train-set BEST (CSV: 2_{index+2}) (C={best_acc_c_test})")
        print(f'SVM train-set BEST (CSV: 2_{index+2}) Confusion matrix: {confusion_matrix(Data_train[:, 2],svc_classifier_granica_TRAIN_BEST.predict(Data_train[:, 0:2]))}')
        
        svc_classifier_granica_TRAIN_WORST = SVC(kernel='linear', C=worst_acc_c_train)
        svc_classifier_granica_TRAIN_WORST.fit(Data_train[:, 0:2], Data_train[:, 2])
        plot_decision_boundary_ax(Data_train[:,0:2], axes_dec=svm_ax[1, 0], func=lambda X: svc_classifier_granica_TRAIN_WORST.predict(X))
        svm_ax[1, 0].set_title(f"SVM train-set WORST (CSV: 2_{index+2}) (C={worst_acc_c_train})")
        print(f'SVM train-set WORST (CSV: 2_{index+2}) Confusion matrix: {confusion_matrix(Data_train[:, 2],svc_classifier_granica_TRAIN_WORST.predict(Data_train[:, 0:2]))}')
        
        svc_classifier_granica_TRAIN_MAX = SVC(kernel='linear', C=max_accuracy_train)
        svc_classifier_granica_TRAIN_MAX.fit(Data_train[:, 0:2], Data_train[:, 2])
        plot_decision_boundary_ax(Data_train[:,0:2], axes_dec=svm_ax[2, 0], func=lambda X: svc_classifier_granica_TRAIN_MAX.predict(X))
        svm_ax[2, 0].set_title(f"SVM train-set MAX (CSV: 2_{index+2}) (C={max_acc_c_train})")
        print(f'SVM train-set MAX (CSV: 2_{index+2}) Confusion matrix: {confusion_matrix(Data_train[:, 2],svc_classifier_granica_TRAIN_MAX.predict(Data_train[:, 0:2]))}')
         
        #Test
        svc_classifier_granica_TEST_BEST = SVC(kernel='linear', C=best_acc_c_test)
        svc_classifier_granica_TEST_BEST.fit(Data_train[:, 0:2], Data_train[:, 2])
        plot_decision_boundary_ax(Data_test[:,0:2], axes_dec=svm_ax[0, 1], func=lambda X: svc_classifier_granica_TEST_BEST.predict(X))
        svm_ax[0, 1].set_title(f"SVM test-set BEST (CSV: 2_{index+2}) (C={best_acc_c_test})")
        print(f'SVM test-set BEST (CSV: 2_{index+2}) Confusion matrix: {confusion_matrix(Data_train[:, 2],svc_classifier_granica_TEST_BEST.predict(Data_train[:, 0:2]))}')
         
        svc_classifier_granica_TEST_WORST = SVC(kernel='linear', C=worst_acc_c_test)
        svc_classifier_granica_TEST_WORST.fit(Data_train[:, 0:2], Data_train[:, 2])
        plot_decision_boundary_ax(Data_train[:,0:2], axes_dec=svm_ax[1, 1], func=lambda X: svc_classifier_granica_TEST_WORST.predict(X))
        svm_ax[1, 1].set_title(f"SVM test-set WORST (CSV: 2_{index+2}) (C={worst_acc_c_test})")
        print(f'SVM test-set WORST (CSV: 2_{index+2}) Confusion matrix: {confusion_matrix(Data_train[:, 2],svc_classifier_granica_TEST_WORST.predict(Data_train[:, 0:2]))}')
        
        svc_classifier_granica_TEST_MAX = SVC(kernel='linear', C=max_accuracy_test)
        svc_classifier_granica_TEST_MAX.fit(Data_train[:, 0:2], Data_train[:, 2])
        plot_decision_boundary_ax(Data_train[:,0:2], axes_dec=svm_ax[2, 1], func=lambda X: svc_classifier_granica_TEST_MAX.predict(X))
        svm_ax[2, 1].set_title(f"SVM test-set MAX (CSV: 2_{index+2}) (C={max_acc_c_test})")
        print(f'SVM test-set MAX (CSV: 2_{index+2}) Confusion matrix: {confusion_matrix(Data_train[:, 2],svc_classifier_granica_TEST_MAX.predict(Data_train[:, 0:2]))}')
         
        acc_ax.semilogx(accuracy_plot_c_train, accuracy_plot_acc_train, 'o', color='green', linestyle='solid', linewidth=2, label="Train Data")
        acc_ax.semilogx(accuracy_plot_c_test, accuracy_plot_acc_test,   'o', color='red',   linestyle='solid', linewidth=2, label="Test Data")
        acc_ax.set_xlabel("c_param")
        acc_ax.legend()

        plt.subplots_adjust(hspace=0.6, wspace=0.5)
        plt.show()


def MLP_granica_decyzyjna_accuracy():
    """
    Wartości parametru C powinny się zmieniać wykładniczo, a na wykresie dobrze jest zastosować skalę logarytmiczną
    """
    for index in range(2):
        Data = np.genfromtxt(f"C:\\Users\\Michał\\Documents\\STUDIA\\II stopień, Informatyka Stosowana - inżynieria oprogramowania i uczenie maszynowe\\I sem\\Obliczenia inteligentne\\Projekt 1 - zadanie 2\\2_{index+2}.csv", delimiter=';')
        Data_train = Data[0:60,:]
        Data_test  = Data[260:300,:]
        
        #Zbior testowy
        best_accuracy_test       : float = 0.0
        best_acc_c_test          : int   = 0
        worst_accuracy_test      : float = 1.0
        worst_acc_c_test         : int   = 1
        max_accuracy_test        : float = 0.0

        accuracy_plot_acc_test = []
        accuracy_plot_c_test   = []
        
        #Zbior treningowy
        worst_accuracy_train      : float = 1.0
        worst_acc_c_train         : int   = 1
        max_accuracy_train        : float = 0.0

        accuracy_plot_acc_train  = []
        accuracy_plot_c_train    = []


        for i, hidden_neurons_param in enumerate(hidden_neurons_MLP):
            mlp_classifier = MLP(hidden_layer_sizes=hidden_neurons_param, max_iter=100000, n_iter_no_change=100000, tol=0, solver='sgd', activation='identity')
            mlp_classifier.fit(Data_train[:, 0:2], Data_train[:, 2])

        #Wyliczanie accuracy
            #Test
            temp_labels_test = mlp_classifier.predict(Data_test[:,0:2])
            accuracy_test = accuracy_score(temp_labels_test, Data_test[:,2])

            accuracy_plot_acc_test.append(accuracy_test)
            accuracy_plot_c_test.append(hidden_neurons_param)
            
            #Train
            temp_labels_train = mlp_classifier.predict(Data_train[:,0:2])
            accuracy_train = accuracy_score(temp_labels_train, Data_train[:,2])

            accuracy_plot_acc_train.append(accuracy_train)
            accuracy_plot_c_train.append(hidden_neurons_param)

            #Wyznaczanie najlepszego i najgorszego accuracy
            if accuracy_test > best_accuracy_test:
                best_accuracy_test = accuracy_test
                best_acc_c_test = hidden_neurons_param
               
            if accuracy_train < worst_accuracy_train:
                worst_accuracy_train = accuracy_test
                worst_acc_c_train = hidden_neurons_param
            
            if accuracy_test < worst_accuracy_test:
                worst_accuracy_test = accuracy_test
                worst_acc_c_test = hidden_neurons_param
            
        svm_fig, svm_ax = plt.subplots(3, 2, figsize=(10, 20)) #Train 0, Test 1
        acc_fig, acc_ax = plt.subplots()

#Granice decyzyjne
        
        max_accuracy_test  = max(accuracy_plot_acc_test)
        max_acc_c_test = hidden_neurons_MLP[accuracy_plot_acc_test.index(max_accuracy_test)]
        
        max_accuracy_train = max(accuracy_plot_acc_train)
        max_acc_c_train = hidden_neurons_MLP[accuracy_plot_acc_train.index(max_accuracy_train)]

        #Train
        mlp_classifier_TRAIN_BEST = MLP(hidden_layer_sizes=best_acc_c_test, max_iter=100000, n_iter_no_change=100000, tol=0, solver='sgd', activation='identity')
        mlp_classifier_TRAIN_BEST.fit(Data_train[:, 0:2], Data_train[:, 2])
        plot_decision_boundary_ax(Data_test[:,0:2], axes_dec=svm_ax[0, 0], func=lambda X: mlp_classifier_TRAIN_BEST.predict(X))
        svm_ax[0, 0].set_title(f"MLP train-set BEST (CSV: 2_{index+2}) (hidden_layers={best_acc_c_test})")
        print(f'MLP train-set BEST (CSV: 2_{index+2}) Confusion matrix: {confusion_matrix(Data_train[:, 2],mlp_classifier_TRAIN_BEST.predict(Data_train[:, 0:2]))}')
        
        mlp_classifier_TRAIN_WORST = MLP(hidden_layer_sizes=worst_acc_c_test, max_iter=100000, n_iter_no_change=100000, tol=0, solver='sgd', activation='identity')
        mlp_classifier_TRAIN_WORST.fit(Data_train[:, 0:2], Data_train[:, 2])
        plot_decision_boundary_ax(Data_train[:,0:2], axes_dec=svm_ax[1, 0], func=lambda X: mlp_classifier_TRAIN_WORST.predict(X))
        svm_ax[1, 0].set_title(f"MLP train-set WORST (CSV: 2_{index+2}) (hidden_layers={worst_acc_c_train})")
        print(f'MLP train-set WORST (CSV: 2_{index+2}) Confusion matrix: {confusion_matrix(Data_train[:, 2],mlp_classifier_TRAIN_WORST.predict(Data_train[:, 0:2]))}')
        
        mlp_classifier_TRAIN_MAX = MLP(hidden_layer_sizes=max_acc_c_train, max_iter=100000, n_iter_no_change=100000, tol=0, solver='sgd', activation='identity')
        mlp_classifier_TRAIN_MAX.fit(Data_train[:, 0:2], Data_train[:, 2])
        plot_decision_boundary_ax(Data_train[:,0:2], axes_dec=svm_ax[2, 0], func=lambda X: mlp_classifier_TRAIN_MAX.predict(X))
        svm_ax[2, 0].set_title(f"MLP train-set MAX (CSV: 2_{index+2}) (hidden_layers={max_acc_c_train})")
        print(f'MLP train-set MAX (CSV: 2_{index+2}) Confusion matrix: {confusion_matrix(Data_train[:, 2],mlp_classifier_TRAIN_MAX.predict(Data_train[:, 0:2]))}')
         
        #Test
        mlp_classifier_TEST_BEST = MLP(hidden_layer_sizes=best_acc_c_test, max_iter=100000, n_iter_no_change=100000, tol=0, solver='sgd', activation='identity')
        mlp_classifier_TEST_BEST.fit(Data_train[:, 0:2], Data_train[:, 2])
        plot_decision_boundary_ax(Data_test[:,0:2], axes_dec=svm_ax[0, 1], func=lambda X: mlp_classifier_TEST_BEST.predict(X))
        svm_ax[0, 1].set_title(f"MLP test-set BEST (CSV: 2_{index+2}) (hidden_layers={best_acc_c_test})")
        print(f'MLP test-set BEST (CSV: 2_{index+2}) Confusion matrix: {confusion_matrix(Data_train[:, 2],mlp_classifier_TEST_BEST.predict(Data_train[:, 0:2]))}')
         
        mlp_classifier_TEST_WORST = MLP(hidden_layer_sizes=worst_acc_c_test, max_iter=100000, n_iter_no_change=100000, tol=0, solver='sgd', activation='identity')
        mlp_classifier_TEST_WORST.fit(Data_train[:, 0:2], Data_train[:, 2])
        plot_decision_boundary_ax(Data_train[:,0:2], axes_dec=svm_ax[1, 1], func=lambda X: mlp_classifier_TEST_WORST.predict(X))
        svm_ax[1, 1].set_title(f"MLP test-set WORST (CSV: 2_{index+2}) (hidden_layers={worst_acc_c_test})")
        print(f'MLP test-set WORST (CSV: 2_{index+2}) Confusion matrix: {confusion_matrix(Data_train[:, 2],mlp_classifier_TEST_WORST.predict(Data_train[:, 0:2]))}')
        
        mlp_classifier_TRAIN_MAX = MLP(hidden_layer_sizes=max_acc_c_test, max_iter=100000, n_iter_no_change=100000, tol=0, solver='sgd', activation='identity')
        mlp_classifier_TRAIN_MAX.fit(Data_train[:, 0:2], Data_train[:, 2])
        plot_decision_boundary_ax(Data_train[:,0:2], axes_dec=svm_ax[2, 1], func=lambda X: mlp_classifier_TRAIN_MAX.predict(X))
        svm_ax[2, 1].set_title(f"MLP test-set MAX (CSV: 2_{index+2}) (hidden_layers={max_acc_c_test})")
        print(f'MLP test-set MAX (CSV: 2_{index+2}) Confusion matrix: {confusion_matrix(Data_train[:, 2],mlp_classifier_TRAIN_MAX.predict(Data_train[:, 0:2]))}')
         
        acc_ax.plot(accuracy_plot_c_train, accuracy_plot_acc_train, 'o', color='green', linestyle='solid', linewidth=2, label="Train Data")
        acc_ax.plot(accuracy_plot_c_test, accuracy_plot_acc_test,   'o', color='red',   linestyle='solid', linewidth=2, label="Test Data")
        acc_ax.set_xlabel("hidden_layer_size")
        acc_ax.legend()

        plt.subplots_adjust(hspace=0.6, wspace=0.5)
        plt.show()


if __name__ == "__main__":
    #KNN_granica_decyzyjna_accuracy()
    #SVM_granica_decyzyjna_accuracy()
    MLP_granica_decyzyjna_accuracy()
    