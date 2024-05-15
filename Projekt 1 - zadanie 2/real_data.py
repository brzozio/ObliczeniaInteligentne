import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier as MLP
from sklearn.neighbors import KNeighborsClassifier as knn
from voronoi import plot_decision_boundary, plot_decision_boundary_ax
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from joblib import dump as model_dump
from joblib import load as model_load
import seaborn as sb
from sklearn.svm import SVC

train_accuracies_all = []
test_accuracies_all  = []
knn_n_neighbours: np.array = [1,2,3,4,5,6,7,8,9,10,11,12,13,14]
#knn_n_neighbours: np.array = [0.1,0.2,0.35,0.4,0.5,0.55,0.6,0.7,0.8,0.9,1.2,1.5,2.1,2.2]


def train_and_evaluate(X_train, X_test, y_train, y_test, name, run, random_state=None) -> tuple[list, list, float, int, float, float]:
    model = MLP(hidden_layer_sizes=10, max_iter=100000, random_state=(random_state+1)*2-1, solver='sgd', tol=0, n_iter_no_change=1000, activation='relu')
    train_accuracies = []
    test_accuracies  = []
    best_epoch_num : int   = 0
    best_epoch_acc : float = 0.0

    for epoch in range(10_000):  
        model.partial_fit(X_train, y_train, classes=np.unique(y_train))
        train_accuracy = accuracy_score(y_train, model.predict(X_train))
        test_accuracy  = accuracy_score(y_test, model.predict(X_test))
        if epoch is 0: 
            model_dump(model, f'mlp_moodel_exp_{name}_EPOCH_0.joblib')
            start_acc_test  = test_accuracy
            start_acc_train = train_accuracy
        if test_accuracy > best_epoch_acc:
            best_epoch_acc = test_accuracy
            best_epoch_num = epoch
            model_dump(model, f'mlp_model_exp_{name}_BEST.joblib')
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)
        print(f"Epoch: {epoch}, Test Accuracy={test_accuracy}, Train Accuracy={train_accuracy} --> {name}, Run: [{run+1}]")
    model_dump(model, f'mlp_moodel_exp_{name}_EPOCH_LAST.joblib')
    end_acc_test  = test_accuracy
    end_acc_train = train_accuracy

    return train_accuracies, test_accuracies, best_epoch_acc, best_epoch_num, start_acc_test, start_acc_train, end_acc_test, end_acc_train

def run_random_state(num_runs, X_train, y_train, X_test, y_test, name)->None:
    df_test = pd.DataFrame({
        'Run': [],
        'AccStart': [],
        'AccBest': [],
        'AccBestEpochNr': [],
        'AccEnd': []
    })
    df_train = pd.DataFrame({
        'Run': [],
        'AccStart': [],
        'AccBest': [],
        'AccBestEpochNr': [],
        'AccEnd': []
    })
    for run in range(num_runs):
        train_accuracies, test_accuracies, best_epoch_acc, best_epoch_num, start_acc_test, start_acc_train, end_acc_test, end_acc_train = train_and_evaluate(X_train, X_test, y_train, y_test, name=name, run=run, random_state=run)
        train_accuracies_all.append(train_accuracies)
        test_accuracies_all.append(test_accuracies)
        
        #if run is 9:
        #plt.figure(figsize=(10, 6))
        #plt.semilogx(range(1, len(train_accuracies_all[run]) + 1), train_accuracies, label=f"Run {run+1} Train")
        #plt.semilogx(range(1, len(test_accuracies_all[run]) + 1), test_accuracies, label=f"Run {run+1} Test")
        #plt.title(f'Acc. Changes Over Epochs - Exp. {name} Data, Run={run+1}')
        #plt.xlabel('Epoch')
        #plt.ylabel('Accuracy')
        #plt.legend()
        #plt.show()
             
        new_row_test = {'Run': run+1, 'AccStart': start_acc_test, 'AccBest': best_epoch_acc, 'AccBestEpochNr': best_epoch_num, 'AccEnd': end_acc_test}
        df_test = df_test._append(new_row_test, ignore_index=True)
        print(df_test)
        
        new_row_train = {'Run': run+1, 'AccStart': start_acc_train, 'AccBest': best_epoch_acc, 'AccBestEpochNr': best_epoch_num, 'AccEnd': end_acc_train}
        df_train = df_train._append(new_row_train, ignore_index=True)
        
    df_test.to_csv(f"test_exp_4_table_exp_{name}_data.csv",   index=False)
    df_train.to_csv(f"train_exp_4_table_exp_{name}_data.csv", index=False)

def KNN_accuracy( X_train, y_train, X_test, y_test):
    #Zbior testowy
    best_accuracy_test       : float = 0.0
    best_acc_n_neighb_test   : int   = 0
    
    accuracy_plot_acc_test          = []
    accuracy_plot_n_neighbour_test  = []


    accuracy_plot_acc_train          = []
    accuracy_plot_n_neighbour_train  = []


    for i, n_neighbours_param in enumerate(knn_n_neighbours):
        knn_classifier = knn(n_neighbors=n_neighbours_param)
        #knn_classifier = SVC(kernel='rbf', C=n_neighbours_param)

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
            
    conf_matrix_fig, conf_ax = plt.subplots(3, 2, figsize=(10, 20)) #Train 0, Test 1
    acc_fig, acc_ax = plt.subplots()

    acc_ax.plot(accuracy_plot_n_neighbour_train, accuracy_plot_acc_train, 'o', color='green', linestyle='solid', linewidth=2, label="Train Data")
    acc_ax.plot(accuracy_plot_n_neighbour_test, accuracy_plot_acc_test,   'o', color='red',   linestyle='solid', linewidth=2, label="Test Data")
    acc_ax.set_xlabel("n_neighbours")
    acc_ax.legend()

    plt.subplots_adjust(hspace=0.6, wspace=0.5)
    knn_classifier_granica_BEST  = knn(n_neighbors=best_acc_n_neighb_test) #zgodnie z wymaganiami najlepszy n-neighbour na podstawie acc ze zbioru testowego
    #knn_classifier_granica_BEST  = SVC(kernel='rbf', C=best_acc_n_neighb_test)
    knn_classifier_granica_BEST.fit(X_train, y_train)
    
    #knn_classifier_granica_WORST = SVC(kernel='rbf', C=knn_n_neighbours[0])
    knn_classifier_granica_WORST = knn(n_neighbors=knn_n_neighbours[0])
    knn_classifier_granica_WORST.fit(X_train, y_train)
    
    #knn_classifier_granica_MAX = SVC(kernel='rbf', C=knn_n_neighbours[13])
    knn_classifier_granica_MAX = knn(n_neighbors=knn_n_neighbours[13])
    knn_classifier_granica_MAX.fit(X_train, y_train)
    plt.figure(figsize=(10, 7))        
    sb.heatmap(confusion_matrix(y_train,knn_classifier_granica_BEST.predict(X_train)), annot=True, cmap='Blues', fmt='g', ax=conf_ax[0,0])
    sb.heatmap(confusion_matrix(y_train,knn_classifier_granica_WORST.predict(X_train)), annot=True, cmap='Blues', fmt='g', ax=conf_ax[1,0])
    sb.heatmap(confusion_matrix(y_train,knn_classifier_granica_MAX.predict(X_train)), annot=True, cmap='Blues', fmt='g', ax=conf_ax[2,0])
    
    
    sb.heatmap(confusion_matrix(y_test,knn_classifier_granica_BEST.predict(X_test)), annot=True, cmap='Blues', fmt='g', ax=conf_ax[0,1])
    sb.heatmap(confusion_matrix(y_test,knn_classifier_granica_WORST.predict(X_test)), annot=True, cmap='Blues', fmt='g', ax=conf_ax[1,1])
    sb.heatmap(confusion_matrix(y_test,knn_classifier_granica_MAX.predict(X_test)), annot=True, cmap='Blues', fmt='g', ax=conf_ax[2,1])
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.show()

if __name__ == "__main__":
    Data = np.genfromtxt(f"C:\\Users\\Michał\\Documents\\STUDIA\\II stopień, Informatyka Stosowana - inżynieria oprogramowania i uczenie maszynowe\\I sem\\Obliczenia inteligentne\\Projekt 1 - zadanie 2\\2_3.csv", delimiter=';')

    num_runs = 10
#Iris
    iris                = datasets.load_iris()
    iris.data  = StandardScaler().fit_transform(iris.data)
    #iris.data           = StandardScaler().fit_transform(iris.data)
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.25, train_size=0.75)
    run_random_state(num_runs=num_runs, name='iris', X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    #KNN_accuracy(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
#Wine
    wine                = datasets.load_wine()
    wine.data  = StandardScaler().fit_transform(wine.data)
    #wine.data           = StandardScaler().fit_transform(wine.data)
    X_train, X_test, y_train, y_test = train_test_split(wine.data,wine.target, test_size=0.3, train_size=0.7)
    run_random_state(num_runs=num_runs, name='wine', X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    #KNN_accuracy(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
#Breast
    breast_cancer       = datasets.load_breast_cancer()
    breast_cancer.data  = StandardScaler().fit_transform(breast_cancer.data)
    X_train, X_test, y_train, y_test = train_test_split(breast_cancer.data, breast_cancer.target, test_size=0.25, train_size=0.75)
    run_random_state(num_runs=num_runs, name='breast', X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    #KNN_accuracy(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)


    
    
       

    