import datasets_get
import numpy as np
import seaborn as sb
import pandas as pd
import torch 
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch import load as load_model
from torch import save as save_model
from model import MLP
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, silhouette_score
from joblib import dump, load


#flattened_mnist_tsne = load(f'flattened_mnist_tsne_afterTransform_{testtrain}.joblib') 
#dump(flattened_mnist_tsne, f'flattened_mnist_tsne_afterTransform_{testtrain}.joblib') 
hidden_neurons_iris          = [1,2,3,4,5,7,10]
hidden_neurons_wine          = [1,2,3,5,7,14,20]
hidden_neurons_breast_cancer = [1,2,6,15,24,30,40]

hidden_neurons_flatten      = [5,10,392,784,1176]
hidden_neurons_PCA          = [1,2,5,10,15]
hidden_neurons_TSNE         = [1,2,5,10,15]
hidden_neurons_3            = [5,8,10,15,20]
hidden_neurons_4            = [5,10,28,56,84]

num_epochs: int      = 1_000
acc_step  : int      = 5

def file_creation_3_datasets() -> None:
    accuracy_score_list = np.zeros(int(num_epochs/acc_step))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #data_set, features_size, class_size, data_name, hidden_neurons = datasets_get.iris(device)
    #data_set, features_size, class_size, data_name, hidden_neurons = datasets_get.wine(device)
    data_set, features_size, class_size, data_name, hidden_neurons = datasets_get.breast_cancer(device)

    X_train, X_test, y_train, y_test = train_test_split(data_set.data, data_set.targets, test_size=0.2,  random_state=42)

    for neuron_i in range(7):
        if data_name is 'iris':
            model = MLP(input_size=features_size, hidden_layer_size=hidden_neurons_iris[neuron_i], classes=class_size)
            neuron = hidden_neurons_iris[neuron_i]
        elif data_name is 'wine':
            model = MLP(input_size=features_size, hidden_layer_size=hidden_neurons_wine[neuron_i], classes=class_size)
            neuron = hidden_neurons_wine[neuron_i]
        elif data_name is 'breast_cancer':
            model = MLP(input_size=features_size, hidden_layer_size=hidden_neurons_breast_cancer[neuron_i], classes=class_size)
            neuron = hidden_neurons_breast_cancer[neuron_i]

        criteria = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01)

        model.to(device)
    
        data_set.data    = X_train
        data_set.targets = y_train
        model.train()
        model.double()

        for epoch in range(num_epochs):
            outputs = model(data_set.data)
            loss = criteria(outputs, data_set.targets)
            optimizer.zero_grad()   # Zerowanie gradientów, aby uniknąć akumulacji w kolejnych krokach
            loss.backward()         # Backpropagation: Obliczenie gradientów
            optimizer.step()        # Aktualizacja wag
            
            if epoch % acc_step == 0:
                model.eval()
                outputs = model(X_test)
                predicted_classes = torch.argmax(outputs, dim=1)
                accuracy_score_list[int(epoch/acc_step)] = accuracy_score(predicted_classes.cpu(), y_test.cpu())
                model.train()

        dump(accuracy_score_list, f'accuracy_score_{data_name}_{neuron}.joblib') 

def chart_3_datasets() -> None:
    fig, ax = plt.subplots(3)
    list_names = ['iris', 'wine', 'breast_cancer']
    for name_i, name in enumerate(list_names):
        for neuron_i in range(7):
            if name is 'iris':
                neuron = hidden_neurons_iris[neuron_i]
            elif name is 'wine':
                neuron = hidden_neurons_wine[neuron_i]
            elif name is 'breast_cancer':
                neuron = hidden_neurons_breast_cancer[neuron_i]

            job_object = load(f'accuracy_score_{name}_{neuron}.joblib')
            ax[name_i].plot(range(int(num_epochs/acc_step)), job_object, label=neuron)
        
        ax[name_i].set_title(name)
        ax[name_i].legend()
    
    plt.xlabel("Num Epochs")
    plt.ylabel("Accuracy Score")
    plt.show()

def file_creation_mnist() -> None:
    accuracy_score_list = np.zeros(int(num_epochs/acc_step))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #Pobieranie danych testowych zbioru MNIST
    data_set_test, features_size, class_size, data_name, hidden_neurons = datasets_get.mnist_flatten(device, False)
    #data_set_test, features_size, class_size, data_name, hidden_neurons = datasets_get.mnist_extr_PCA(device, False)
    #data_set_test, features_size, class_size, data_name, hidden_neurons = datasets_get.mnist_extr_TSNE(device, False, 'test')
    #data_set_test, features_size, class_size, data_name, hidden_neurons = datasets_get.mnist_extr_3(device,  False, 'test')
    #data_set_test, features_size, class_size, data_name, hidden_neurons = datasets_get.mnist_extr_4(device, False, 'test')
    
    #Pobieranie danych treningowych zbioru MNIST
    data_set_train, features_size, class_size, data_name, hidden_neurons = datasets_get.mnist_flatten(device, True)
    #data_set_train, features_size, class_size, data_name, hidden_neurons = datasets_get.mnist_extr_PCA(device, True)
    #data_set_train, features_size, class_size, data_name, hidden_neurons = datasets_get.mnist_extr_TSNE(device, True, 'train')
    #data_set_train, features_size, class_size, data_name, hidden_neurons = datasets_get.mnist_extr_3(device,  True, 'train' )
    #data_set_train, features_size, class_size, data_name, hidden_neurons = datasets_get.mnist_extr_4(device, True, 'train')
    
    for neuron_i in range(5):
        print(f"DATA NAME: {data_name}")
        if data_name is 'mnist_extr_3':
            model = MLP(input_size=features_size, hidden_layer_size=hidden_neurons_3[neuron_i], classes=class_size)
            neuron = hidden_neurons_3[neuron_i]
        elif data_name is 'mnist_extr_4':
            model = MLP(input_size=features_size, hidden_layer_size=hidden_neurons_4[neuron_i], classes=class_size)
            neuron = hidden_neurons_4[neuron_i]
        elif data_name is 'mnist_2_features_TSNE':
            model = MLP(input_size=features_size, hidden_layer_size=hidden_neurons_TSNE[neuron_i], classes=class_size)
            neuron = hidden_neurons_TSNE[neuron_i]
        elif data_name is 'mnist_2_features_PCA':
            model = MLP(input_size=features_size, hidden_layer_size=hidden_neurons_PCA[neuron_i], classes=class_size)
            neuron = hidden_neurons_PCA[neuron_i]
        elif data_name is 'mnist_flatten':
            model = MLP(input_size=features_size, hidden_layer_size=hidden_neurons_flatten[neuron_i], classes=class_size)
            neuron = hidden_neurons_flatten[neuron_i]

        criteria = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01)

        model.to(device)
    
        model.train()
        model.double()

        for epoch in range(num_epochs):
            outputs = model(data_set_train.data)
            loss = criteria(outputs, data_set_train.targets)
            optimizer.zero_grad()   # Zerowanie gradientów, aby uniknąć akumulacji w kolejnych krokach
            loss.backward()         # Backpropagation: Obliczenie gradientów
            optimizer.step()        # Aktualizacja wag
            
            if epoch % acc_step == 0:
                model.eval()
                outputs = model(data_set_test.data)
                predicted_classes = torch.argmax(outputs, dim=1)
                accuracy_score_list[int(epoch/acc_step)] = accuracy_score(predicted_classes.cpu(), data_set_test.targets.cpu())
                model.train()
                print(f'Epoch: {epoch}')

        dump(accuracy_score_list, f'accuracy_score_{data_name}_{neuron}.joblib') 

def chart_3_datasets() -> None:
    fig, ax = plt.subplots(3)
    list_names = ['iris', 'wine', 'breast_cancer']
    for name_i, name in enumerate(list_names):
        for neuron_i in range(7):
            if name is 'iris':
                neuron = hidden_neurons_iris[neuron_i]
            elif name is 'wine':
                neuron = hidden_neurons_wine[neuron_i]
            elif name is 'breast_cancer':
                neuron = hidden_neurons_breast_cancer[neuron_i]

            job_object = load(f'accuracy_score_{name}_{neuron}.joblib')
            ax[name_i].plot(range(int(num_epochs/acc_step)), job_object, label=neuron)
        
        ax[name_i].set_title(name)
        ax[name_i].legend()
    
    plt.xlabel("Num Epochs")
    plt.ylabel("Accuracy Score")
    plt.show()

if __name__ == "__main__":
   #file_creation_3_datasets()
   #chart_3_datasets()
   file_creation_mnist()

