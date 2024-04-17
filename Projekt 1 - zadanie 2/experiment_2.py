import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.neural_network import MLPClassifier as MLP
from sklearn.svm import SVC
from voronoi import plot_decision_boundary, plot_decision_boundary_ax
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import seaborn as sb


knn_n_neighbours: np.array = [1,2,3,4,5,6,7,8,9,10,11,12,13,14]
c_parameter_SVM   : np.array = [0.001,0.01,0.1,1,10,100,1000,10000]
hidden_neurons_MLP: np.array = [2,6,10,30]


def KNN_granica_decyzyjna_accuracy():
    for index in range(2):
        Data = np.genfromtxt(f"C:\\Users\\Michał\\Documents\\STUDIA\\II stopień, Informatyka Stosowana - inżynieria oprogramowania i uczenie maszynowe\\I sem\\Obliczenia inteligentne\\Projekt 1 - zadanie 2\\2_{index+2}.csv", delimiter=';')
        #Data_train = Data[0:260,:]
        #Data_test  = Data[260:300,:]
        X_train, X_test, y_train, y_test = train_test_split(Data[:,0:2], Data[:,2], test_size=0.2, random_state=42)
        
        #Zbior testowy
        best_accuracy_test       : float = 0.0
        best_acc_n_neighb_test   : int   = 0
        
        accuracy_plot_acc_test          = []
        accuracy_plot_n_neighbour_test  = []
    

        accuracy_plot_acc_train          = []
        accuracy_plot_n_neighbour_train  = []


        for i, n_neighbours_param in enumerate(knn_n_neighbours):
            knn_classifier = knn(n_neighbors=n_neighbours_param)

            knn_classifier.fit(X_train, y_train)

        #Wyliczanie accuracy
            #Test
            temp_labels_test = knn_classifier.predict(X_test)
            accuracy_test = accuracy_score(temp_labels_test, y_test)

            accuracy_plot_acc_test.append(accuracy_test)
            accuracy_plot_n_neighbour_test.append(n_neighbours_param)
            
            #Train
            temp_labels_train = knn_classifier.predict(X_train)
            accuracy_train = accuracy_score(temp_labels_train, y_train)

            accuracy_plot_acc_train.append(accuracy_train)
            accuracy_plot_n_neighbour_train.append(n_neighbours_param)

            #Wyznaczanie najlepszego i najgorszego accuracy
            if accuracy_test > best_accuracy_test:
                best_accuracy_test = accuracy_test
                best_acc_n_neighb_test = n_neighbours_param
               
            
        knn_fig, knn_ax = plt.subplots(3, 2, figsize=(10, 20)) #Train 0, Test 1
        conf_matrix_fig, conf_ax = plt.subplots(3, 2, figsize=(10, 20)) #Train 0, Test 1
        acc_fig, acc_ax = plt.subplots()

#Granice decyzyjne
        #Train
        knn_classifier_granica_BEST  = knn(n_neighbors=best_acc_n_neighb_test) #zgodnie z wymaganiami najlepszy n-neighbour na podstawie acc ze zbioru testowego
        knn_classifier_granica_BEST.fit(X_train, y_train)
        knn_classifier_granica_WORST = knn(n_neighbors=knn_n_neighbours[0])
        knn_classifier_granica_WORST.fit(X_train, y_train)
        knn_classifier_granica_MAX = knn(n_neighbors=knn_n_neighbours[13])
        knn_classifier_granica_MAX.fit(X_train, y_train)
        
        plot_decision_boundary_ax(X_train, axes_dec=knn_ax[0, 0], func=lambda X: knn_classifier_granica_BEST.predict(X))
        knn_ax[0, 0].set_title(f"KNN train-set BEST (CSV: 2_{index+2}) (Neighbours={best_acc_n_neighb_test})")
        print(f'KNN train-set BEST (CSV: 2_{index+2}) Confusion matrix: {confusion_matrix(y_train,knn_classifier_granica_BEST.predict(X_train))}')
        
        plot_decision_boundary_ax(X_train, axes_dec=knn_ax[1, 0], func=lambda X: knn_classifier_granica_WORST.predict(X))
        knn_ax[1, 0].set_title(f"KNN train-set MIN (CSV: 2_{index+2}) (Neighbours={knn_n_neighbours[0]})")
        print(f'KNN train-set WORST (CSV: 2_{index+2}) Confusion matrix: {confusion_matrix(y_train,knn_classifier_granica_WORST.predict(X_train))}')
        
        plot_decision_boundary_ax(X_train, axes_dec=knn_ax[2, 0], func=lambda X: knn_classifier_granica_MAX.predict(X))
        knn_ax[2, 0].set_title(f"KNN train-set MAX (CSV: 2_{index+2}) (Neighbours={knn_n_neighbours[13]})")
        print(f'KNN train-set MAX (CSV: 2_{index+2}) Confusion matrix: {confusion_matrix(y_train,knn_classifier_granica_MAX.predict(X_train))}')
         
        #Test
        plot_decision_boundary_ax(X_test, axes_dec=knn_ax[0, 1], func=lambda X: knn_classifier_granica_BEST.predict(X))
        knn_ax[0, 1].set_title(f"KNN test-set BEST (CSV: 2_{index+2}) (Neighbours={best_acc_n_neighb_test})")
        print(f'KNN test-set BEST (CSV: 2_{index+2}) Confusion matrix: {confusion_matrix(y_test,knn_classifier_granica_BEST.predict(X_test))}')
         
        plot_decision_boundary_ax(X_test, axes_dec=knn_ax[1, 1], func=lambda X: knn_classifier_granica_WORST.predict(X))
        knn_ax[1, 1].set_title(f"KNN test-set MIN (CSV: 2_{index+2}) (Neighbours={knn_n_neighbours[0]})")
        print(f'KNN test-set WORST (CSV: 2_{index+2}) Confusion matrix: {confusion_matrix(y_test,knn_classifier_granica_WORST.predict(X_test))}')
  
        plot_decision_boundary_ax(X_test, axes_dec=knn_ax[2, 1], func=lambda X: knn_classifier_granica_MAX.predict(X))
        knn_ax[2, 1].set_title(f"KNN test-set MAX (CSV: 2_{index+2}) (Neighbours={knn_n_neighbours[13]})")
        print(f'KNN test-set MAX (CSV: 2_{index+2}) Confusion matrix: {confusion_matrix(y_test,knn_classifier_granica_MAX.predict(X_test))}')
         
        acc_ax.plot(accuracy_plot_n_neighbour_train, accuracy_plot_acc_train, 'o', color='green', linestyle='solid', linewidth=2, label="Train Data")
        acc_ax.plot(accuracy_plot_n_neighbour_test, accuracy_plot_acc_test,   'o', color='red',   linestyle='solid', linewidth=2, label="Test Data")
        acc_ax.set_xlabel("n_neighbours")
        acc_ax.legend()

        plt.subplots_adjust(hspace=0.6, wspace=0.5)

        plt.figure(figsize=(10, 7))        
        sb.heatmap(confusion_matrix(y_train,knn_classifier_granica_BEST.predict(X_train)), annot=True, cmap='Blues', fmt='g', ax=conf_ax[0,0])
        sb.heatmap(confusion_matrix(y_train,knn_classifier_granica_WORST.predict(X_train)), annot=True, cmap='Blues', fmt='g', ax=conf_ax[1,0])
        sb.heatmap(confusion_matrix(y_train,knn_classifier_granica_MAX.predict(X_train)), annot=True, cmap='Blues', fmt='g', ax=conf_ax[2,0])
        
        
        sb.heatmap(confusion_matrix(y_train,knn_classifier_granica_BEST.predict(X_train)), annot=True, cmap='Blues', fmt='g', ax=conf_ax[0,1])
        sb.heatmap(confusion_matrix(y_train,knn_classifier_granica_WORST.predict(X_train)), annot=True, cmap='Blues', fmt='g', ax=conf_ax[1,1])
        sb.heatmap(confusion_matrix(y_train,knn_classifier_granica_MAX.predict(X_train)), annot=True, cmap='Blues', fmt='g', ax=conf_ax[2,1])
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.show()


def SVM_granica_decyzyjna_accuracy():
    """
    Wartości parametru C powinny się zmieniać wykładniczo, a na wykresie dobrze jest zastosować skalę logarytmiczną
    """
    for index in range(2):
        Data = np.genfromtxt(f"C:\\Users\\Michał\\Documents\\STUDIA\\II stopień, Informatyka Stosowana - inżynieria oprogramowania i uczenie maszynowe\\I sem\\Obliczenia inteligentne\\Projekt 1 - zadanie 2\\2_{index+2}.csv", delimiter=';')
        Data_train = Data[0:260,:]
        Data_test  = Data[260:300,:]
        X_train, X_test, y_train, y_test = train_test_split(Data[:,0:2], Data[:,2], test_size=0.2, random_state=42)
        
        #Zbior testowy
        best_accuracy_test       : float = 0.0
        best_acc_c_test          : int   = 0
        accuracy_plot_acc_test = []
        accuracy_plot_c_test   = []
    

        accuracy_plot_acc_train  = []
        accuracy_plot_c_train    = []


        for i, c_param in enumerate(c_parameter_SVM):
            svc_linear_classifier = SVC(kernel='linear', C=c_param)

            svc_linear_classifier.fit(X_train, y_train)

        #Wyliczanie accuracy
            #Test
            temp_labels_test = svc_linear_classifier.predict(X_test)
            accuracy_test = accuracy_score(temp_labels_test, y_test)

            accuracy_plot_acc_test.append(accuracy_test)
            accuracy_plot_c_test.append(c_param)
            
            #Train
            temp_labels_train = svc_linear_classifier.predict(X_train)
            accuracy_train = accuracy_score(temp_labels_train, y_train)

            accuracy_plot_acc_train.append(accuracy_train)
            accuracy_plot_c_train.append(c_param)

            #Wyznaczanie najlepszego i najgorszego accuracy
            if accuracy_test > best_accuracy_test:
                best_accuracy_test = accuracy_test
                best_acc_c_test = c_param
               
            
        svm_fig, svm_ax          = plt.subplots(3, 2, figsize=(10, 20)) #Train 0, Test 1
        conf_matrix_fig, conf_ax = plt.subplots(3, 2, figsize=(10, 20)) #Train 0, Test 1
        acc_fig, acc_ax          = plt.subplots()

#Granice decyzyjne
        #Train
        svc_classifier_granica_BEST = SVC(kernel='linear', C=best_acc_c_test)
        svc_classifier_granica_BEST.fit(X_train, y_train)
        
        svc_classifier_granica_WORST = SVC(kernel='linear', C=c_parameter_SVM[0])
        svc_classifier_granica_WORST.fit(X_train, y_train)
        
        svc_classifier_granica_MAX = SVC(kernel='linear', C=c_parameter_SVM[7])
        svc_classifier_granica_MAX.fit(X_train, y_train)
        
        plot_decision_boundary_ax(X_train, axes_dec=svm_ax[0, 0], func=lambda X: svc_classifier_granica_BEST.predict(X))
        svm_ax[0, 0].set_title(f"SVM train-set BEST (CSV: 2_{index+2}) (C={best_acc_c_test})")
        print(f'SVM train-set BEST (CSV: 2_{index+2}) Confusion matrix: {confusion_matrix(y_train,svc_classifier_granica_BEST.predict(X_train))}')
        
        plot_decision_boundary_ax(X_train, axes_dec=svm_ax[1, 0], func=lambda X: svc_classifier_granica_WORST.predict(X))
        svm_ax[1, 0].set_title(f"SVM train-set MIN (CSV: 2_{index+2}) (C={c_parameter_SVM[0]})")
        print(f'SVM train-set WORST (CSV: 2_{index+2}) Confusion matrix: {confusion_matrix(y_train,svc_classifier_granica_WORST.predict(X_train))}')
        
        plot_decision_boundary_ax(X_train, axes_dec=svm_ax[2, 0], func=lambda X: svc_classifier_granica_MAX.predict(X))
        svm_ax[2, 0].set_title(f"SVM train-set MAX (CSV: 2_{index+2}) (C={c_parameter_SVM[7]})")
        print(f'SVM train-set MAX (CSV: 2_{index+2}) Confusion matrix: {confusion_matrix(y_train,svc_classifier_granica_MAX.predict(X_train))}')
         
        #Test
       
        plot_decision_boundary_ax(X_test, axes_dec=svm_ax[0, 1], func=lambda X: svc_classifier_granica_BEST.predict(X))
        svm_ax[0, 1].set_title(f"SVM test-set BEST (CSV: 2_{index+2}) (C={best_acc_c_test})")
        print(f'SVM test-set BEST (CSV: 2_{index+2}) Confusion matrix: {confusion_matrix(y_test,svc_classifier_granica_BEST.predict(X_test))}')
         
        plot_decision_boundary_ax(X_test, axes_dec=svm_ax[1, 1], func=lambda X: svc_classifier_granica_WORST.predict(X))
        svm_ax[1, 1].set_title(f"SVM test-set MIN (CSV: 2_{index+2}) (C={c_parameter_SVM[0]})")
        print(f'SVM test-set WORST (CSV: 2_{index+2}) Confusion matrix: {confusion_matrix(y_test,svc_classifier_granica_WORST.predict(X_test))}')
       
        plot_decision_boundary_ax(X_test, axes_dec=svm_ax[2, 1], func=lambda X: svc_classifier_granica_MAX.predict(X))
        svm_ax[2, 1].set_title(f"SVM test-set MAX (CSV: 2_{index+2}) (C={c_parameter_SVM[7]})")
        print(f'SVM test-set MAX (CSV: 2_{index+2}) Confusion matrix: {confusion_matrix(y_test,svc_classifier_granica_MAX.predict(X_test))}')
         
        acc_ax.semilogx(accuracy_plot_c_train, accuracy_plot_acc_train, 'o', color='green', linestyle='solid', linewidth=2, label="Train Data")
        acc_ax.semilogx(accuracy_plot_c_test, accuracy_plot_acc_test,   'o', color='red',   linestyle='solid', linewidth=2, label="Test Data")
        acc_ax.set_xlabel("c_param")
        acc_ax.legend()
        
        plt.subplots_adjust(hspace=0.6, wspace=0.5)
        plt.show()
        

        plt.figure(figsize=(10, 7))        
        sb.heatmap(confusion_matrix(y_train,svc_classifier_granica_BEST.predict(X_train)), annot=True, cmap='Blues', fmt='g', ax=conf_ax[0,0])
        sb.heatmap(confusion_matrix(y_train,svc_classifier_granica_WORST.predict(X_train)), annot=True, cmap='Blues', fmt='g', ax=conf_ax[1,0])
        sb.heatmap(confusion_matrix(y_train,svc_classifier_granica_MAX.predict(X_train)), annot=True, cmap='Blues', fmt='g', ax=conf_ax[2,0])
        
        
        sb.heatmap(confusion_matrix(y_train,svc_classifier_granica_BEST.predict(X_train)), annot=True, cmap='Blues', fmt='g', ax=conf_ax[0,1])
        sb.heatmap(confusion_matrix(y_train,svc_classifier_granica_WORST.predict(X_train)), annot=True, cmap='Blues', fmt='g', ax=conf_ax[1,1])
        sb.heatmap(confusion_matrix(y_train,svc_classifier_granica_MAX.predict(X_train)), annot=True, cmap='Blues', fmt='g', ax=conf_ax[2,1])
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.show()

        





def MLP_granica_decyzyjna_accuracy():
    """
    Wartości parametru C powinny się zmieniać wykładniczo, a na wykresie dobrze jest zastosować skalę logarytmiczną
    """
    for index in range(2):
        Data = np.genfromtxt(f"C:\\Users\\Michał\\Documents\\STUDIA\\II stopień, Informatyka Stosowana - inżynieria oprogramowania i uczenie maszynowe\\I sem\\Obliczenia inteligentne\\Projekt 1 - zadanie 2\\2_{index+2}.csv", delimiter=';')
        Data_train = Data[0:260,:]
        Data_test  = Data[260:300,:]
        X_train, X_test, y_train, y_test = train_test_split(Data[:,0:2], Data[:,2], test_size=0.2, random_state=42)
        
        #Zbior testowy
        best_accuracy_test       : float = 0.0
        best_acc_c_test          : int   = 0

        accuracy_plot_acc_test = []
        accuracy_plot_c_test   = []


        accuracy_plot_acc_train  = []
        accuracy_plot_c_train    = []


        for i, hidden_neurons_param in enumerate(hidden_neurons_MLP):
            mlp_classifier = MLP(hidden_layer_sizes=hidden_neurons_param, max_iter=100000, n_iter_no_change=100000, tol=0, solver='sgd', activation='identity')
            mlp_classifier.fit(X_train, y_train)

        #Wyliczanie accuracy
            #Test
            temp_labels_test = mlp_classifier.predict(X_test)
            accuracy_test = accuracy_score(temp_labels_test, y_test)

            accuracy_plot_acc_test.append(accuracy_test)
            accuracy_plot_c_test.append(hidden_neurons_param)
            
            #Train
            temp_labels_train = mlp_classifier.predict(X_train)
            accuracy_train = accuracy_score(temp_labels_train, y_train)

            accuracy_plot_acc_train.append(accuracy_train)
            accuracy_plot_c_train.append(hidden_neurons_param)

            #Wyznaczanie najlepszego i najgorszego accuracy
            if accuracy_test > best_accuracy_test:
                best_accuracy_test = accuracy_test
                best_acc_c_test = hidden_neurons_param
            
        svm_fig, svm_ax = plt.subplots(3, 2, figsize=(10, 20)) #Train 0, Test 1
        conf_matrix_fig, conf_ax = plt.subplots(3, 2, figsize=(10, 20)) #Train 0, Test 1
        acc_fig, acc_ax = plt.subplots()

#Granice decyzyjne

        #Train
        mlp_classifier_BEST = MLP(hidden_layer_sizes=best_acc_c_test, max_iter=100000, n_iter_no_change=100000, tol=0, solver='sgd', activation='identity')
        mlp_classifier_BEST.fit(X_train, y_train)
        
        mlp_classifier_WORST = MLP(hidden_layer_sizes=hidden_neurons_MLP[0], max_iter=100000, n_iter_no_change=100000, tol=0, solver='sgd', activation='identity')
        mlp_classifier_WORST.fit(X_train, y_train)
        
        mlp_classifier_MAX = MLP(hidden_layer_sizes=hidden_neurons_MLP[3], max_iter=100000, n_iter_no_change=100000, tol=0, solver='sgd', activation='identity')
        mlp_classifier_MAX.fit(X_train, y_train)
        
        plot_decision_boundary_ax(X_train, axes_dec=svm_ax[0, 0], func=lambda X: mlp_classifier_BEST.predict(X))
        svm_ax[0, 0].set_title(f"MLP train-set BEST (CSV: 2_{index+2}) (hidden_layers={best_acc_c_test})")
        print(f'MLP train-set BEST (CSV: 2_{index+2}) Confusion matrix: {confusion_matrix(y_train,mlp_classifier_BEST.predict(X_train))}')
        
        plot_decision_boundary_ax(X_train, axes_dec=svm_ax[1, 0], func=lambda X: mlp_classifier_WORST.predict(X))
        svm_ax[1, 0].set_title(f"MLP train-set MIN (CSV: 2_{index+2}) (hidden_layers={hidden_neurons_MLP[0]})")
        print(f'MLP train-set WORST (CSV: 2_{index+2}) Confusion matrix: {confusion_matrix(y_train,mlp_classifier_WORST.predict(X_train))}')
        
        plot_decision_boundary_ax(X_train, axes_dec=svm_ax[2, 0], func=lambda X: mlp_classifier_MAX.predict(X))
        svm_ax[2, 0].set_title(f"MLP train-set MAX (CSV: 2_{index+2}) (hidden_layers={hidden_neurons_MLP[3]})")
        print(f'MLP train-set MAX (CSV: 2_{index+2}) Confusion matrix: {confusion_matrix(y_train,mlp_classifier_MAX.predict(X_train))}')
         
        #Test
       
        plot_decision_boundary_ax(X_test, axes_dec=svm_ax[0, 1], func=lambda X: mlp_classifier_BEST.predict(X))
        svm_ax[0, 1].set_title(f"MLP test-set BEST (CSV: 2_{index+2}) (hidden_layers={best_acc_c_test})")
        print(f'MLP test-set BEST (CSV: 2_{index+2}) Confusion matrix: {confusion_matrix(y_test,mlp_classifier_BEST.predict(X_test))}')
         
        
        plot_decision_boundary_ax(X_test, axes_dec=svm_ax[1, 1], func=lambda X: mlp_classifier_WORST.predict(X))
        svm_ax[1, 1].set_title(f"MLP test-set MIN (CSV: 2_{index+2}) (hidden_layers={hidden_neurons_MLP[0]})")
        print(f'MLP test-set WORST (CSV: 2_{index+2}) Confusion matrix: {confusion_matrix(y_test,mlp_classifier_WORST.predict(X_test))}')
        
       
        plot_decision_boundary_ax(X_test, axes_dec=svm_ax[2, 1], func=lambda X: mlp_classifier_MAX.predict(X))
        svm_ax[2, 1].set_title(f"MLP test-set MAX (CSV: 2_{index+2}) (hidden_layers={hidden_neurons_MLP[3]})")
        print(f'MLP test-set MAX (CSV: 2_{index+2}) Confusion matrix: {confusion_matrix(y_test,mlp_classifier_MAX.predict(X_test))}')
         
        acc_ax.plot(accuracy_plot_c_train, accuracy_plot_acc_train, 'o', color='green', linestyle='solid', linewidth=2, label="Train Data")
        acc_ax.plot(accuracy_plot_c_test, accuracy_plot_acc_test,   'o', color='red',   linestyle='solid', linewidth=2, label="Test Data")
        acc_ax.set_xlabel("hidden_layer_size")
        acc_ax.legend()
        
        plt.subplots_adjust(hspace=0.6, wspace=0.5)

        plt.figure(figsize=(10, 7))        
        sb.heatmap(confusion_matrix(y_train,mlp_classifier_BEST.predict(X_train)), annot=True, cmap='Blues', fmt='g', ax=conf_ax[0,0])
        sb.heatmap(confusion_matrix(y_train,mlp_classifier_WORST.predict(X_train)), annot=True, cmap='Blues', fmt='g', ax=conf_ax[1,0])
        sb.heatmap(confusion_matrix(y_train,mlp_classifier_MAX.predict(X_train)), annot=True, cmap='Blues', fmt='g', ax=conf_ax[2,0])
        
        
        sb.heatmap(confusion_matrix(y_train,mlp_classifier_BEST.predict(X_train)), annot=True, cmap='Blues', fmt='g', ax=conf_ax[0,1])
        sb.heatmap(confusion_matrix(y_train,mlp_classifier_WORST.predict(X_train)), annot=True, cmap='Blues', fmt='g', ax=conf_ax[1,1])
        sb.heatmap(confusion_matrix(y_train,mlp_classifier_MAX.predict(X_train)), annot=True, cmap='Blues', fmt='g', ax=conf_ax[2,1])
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.show()


if __name__ == "__main__":
    #KNN_granica_decyzyjna_accuracy()
    #SVM_granica_decyzyjna_accuracy()
    MLP_granica_decyzyjna_accuracy()
    