import datasets_get
import numpy as np
import seaborn as sb
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from torch import load as load_model
from torch import save as save_model
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score, silhouette_score
from voronoi import plot_decision_boundary, voronoi
from scipy.spatial import Voronoi
import torch.nn as nn
from model import CNN_tanh, CNN_leaky_relu
import os


def execute_model(data_set, model, batch_size, data_name, num_epochs: int = 200, lr: float = 0.01, train: bool = False, continue_train: bool = False):
    num_epochs            = num_epochs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'CUDA VERSION: {torch.version.cuda}')
    print(f'DEVICE RUNING: {device}')

    criteria = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    if continue_train is True:
        model.load_state_dict(load_model(f'model_{data_name}.pth'))
        
    model.to(device)

    if train is True:
        model.train()
        model.double()
        data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=True)
        print(f'DATA SIZE: {data_set.data.size()}')

        for epoch in range(num_epochs):
            for batch in data_loader:
                data, target = batch['data'].to(device), batch['target'].to(device)
                outputs = model.extract(data)
                outputs = model.forward(outputs)
                loss = criteria(outputs, target)
                
                optimizer.zero_grad()   # Zerowanie gradientów, aby git auniknąć akumulacji w kolejnych krokach
                loss.backward()         # Backpropagation: Obliczenie gradientów
                optimizer.step()        # Aktualizacja wag
            
            print(f"Epoch [{epoch+1}/{num_epochs}]  Loss: {loss.item():.5f}   - {data_name}")

            if epoch % 10 == 0 and epoch != 0:
                save_model(model.state_dict(), f'model_{data_name}.pth') #Zapisz model co 1_000 epok
                print(f"SAVED MODEL: model_{data_name}.pth at epoch [{epoch}]")


        save_model(model.state_dict(), f'model_{data_name}.pth') #Zapisz model na koniec trenignu - koniec epok
    else:

        model.load_state_dict(load_model(f'model_{data_name}.pth'))
        model.eval()
        model.double()
        model.to(device)

        outputs = model.extract(data_set.data)
        outputs = model.forward(outputs)
        print(f"OUTPUS: {outputs}")
        
        softmax = torch.nn.Softmax(dim=1)
        probabilities = softmax(outputs)
        predicted_classes = torch.argmax(probabilities, dim=1)

        print(f'PREDICTED CLASSES: {predicted_classes}')
        print(f"ORIGINAL CLASSES: {data_set.targets}")

        if data_name == 'projekt_2_zad_2_cifar10' or data_name == 'projekt_2_zad_2_cifar10_reduced':
            targets_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
            for index, label in enumerate(predicted_classes):
                mapped_pred_class = targets_names[label]
                orig_class   = data_set.targets[index]
                mapped_orig_class = targets_names[orig_class]
                print(f'PREDICTED: {mapped_pred_class}, {label}, ORIGINAL: {mapped_orig_class}, {orig_class}')

        plt.figure(figsize=(10, 7))        
       
        predicted_classes_cpu = predicted_classes.cpu().numpy()
        targets_cpu           = data_set.targets.cpu().numpy()
        dataset_cpu           = data_set.data.cpu().numpy()
        
        sb.heatmap(confusion_matrix(targets_cpu,predicted_classes_cpu), annot=True, cmap='Blues', fmt='g')
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.title(f"Conf Matrix - {data_name}")
        plt.show()
        
        accuracy = accuracy_score(predicted_classes_cpu, targets_cpu)
        print(f'ACCURACY SCORE FOR {data_name}: {accuracy:.4f}')

        #Diagram Voronoi'a oraz granice decyzyjne dla ekstrakcji do 2 cech
        if model.reduce_to_dim2:

            #plot_decision_boundary(X=data_set.data.cpu(), func=lambda X: model(X), y_true=data_set.targets.cpu())
            data = model.extract(data_set.data).cpu()
            model.to('cpu')

            plot_decision_boundary(X=data, func=lambda X: model.forward(X), tolerance=0.1)
            
            vor = Voronoi(data_set.data.cpu())
            voronoi(vor=vor, etykiety=predicted_classes_cpu)


def execute_model_with_acc_plot(data_set, model, batch_size, data_name, num_epochs: int = 200, lr: float = 0.01, train: bool = False, continue_train: bool = False):
    num_epochs            = num_epochs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'CUDA VERSION: {torch.version.cuda}')
    print(f'DEVICE RUNING: {device}')

    criteria = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    if continue_train is True:
        model.load_state_dict(load_model(f'model_{data_name}.pth'))
        
    model.to(device)

    if train is True:
        model.train()
        model.double()
        data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=True)
        print(f'DATA SIZE: {data_set.data.size()}')

        for epoch in range(num_epochs):
            for batch in data_loader:
                data, target = batch['data'].to(device), batch['target'].to(device)
                outputs = model.extract(data)
                outputs = model.forward(outputs)
                loss = criteria(outputs, target)
                
                optimizer.zero_grad()   # Zerowanie gradientów, aby git auniknąć akumulacji w kolejnych krokach
                loss.backward()         # Backpropagation: Obliczenie gradientów
                optimizer.step()        # Aktualizacja wag
            
            print(f"Epoch [{epoch+1}/{num_epochs}]  Loss: {loss.item():.5f}   - {data_name}")

            if epoch % 10 == 0 and epoch != 0:
                save_model(model.state_dict(), f'model_{data_name}.pth') #Zapisz model co 1_000 epok
                print(f"SAVED MODEL: model_{data_name}.pth at epoch [{epoch}]")


        save_model(model.state_dict(), f'model_{data_name}.pth') #Zapisz model na koniec trenignu - koniec epok
    else:

        model.load_state_dict(load_model(f'model_{data_name}.pth'))
        model.eval()
        model.double()
        model.to(device)

        outputs = model.extract(data_set.data)
        outputs = model.forward(outputs)
        print(f"OUTPUS: {outputs}")
        
        softmax = torch.nn.Softmax(dim=1)
        probabilities = softmax(outputs)
        predicted_classes = torch.argmax(probabilities, dim=1)

        print(f'PREDICTED CLASSES: {predicted_classes}')
        print(f"ORIGINAL CLASSES: {data_set.targets}")

        if data_name == 'projekt_2_zad_2_cifar10' or data_name == 'projekt_2_zad_2_cifar10_reduced':
            targets_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
            for index, label in enumerate(predicted_classes):
                mapped_pred_class = targets_names[label]
                orig_class   = data_set.targets[index]
                mapped_orig_class = targets_names[orig_class]
                print(f'PREDICTED: {mapped_pred_class}, {label}, ORIGINAL: {mapped_orig_class}, {orig_class}')

        plt.figure(figsize=(10, 7))        
       
        predicted_classes_cpu = predicted_classes.cpu().numpy()
        targets_cpu           = data_set.targets.cpu().numpy()
        dataset_cpu           = data_set.data.cpu().numpy()
        
        sb.heatmap(confusion_matrix(targets_cpu,predicted_classes_cpu), annot=True, cmap='Blues', fmt='g')
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.title(f"Conf Matrix - {data_name}")
        plt.show()
        
        accuracy = accuracy_score(predicted_classes_cpu, targets_cpu)
        print(f'ACCURACY SCORE FOR {data_name}: {accuracy:.4f}')

        #Diagram Voronoi'a oraz granice decyzyjne dla ekstrakcji do 2 cech
        if model.reduce_to_dim2:

            #plot_decision_boundary(X=data_set.data.cpu(), func=lambda X: model(X), y_true=data_set.targets.cpu())
            data = model.extract(data_set.data).cpu()
            model.to('cpu')

            plot_decision_boundary(X=data, func=lambda X: model.forward(X), tolerance=0.1)
            
            vor = Voronoi(data_set.data.cpu())
            voronoi(vor=vor, etykiety=predicted_classes_cpu)


def execute_model_fast(data_set_train, data_set_test, model, batch_size, data_name, num_epoch: int = 1600, lr: float = 0.001, calc_interval : int = 16) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'CUDA VERSION: {torch.version.cuda}')
    print(f'DEVICE RUNING: {device}')

    criteria = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
        
    model.to(device)
    model.train()
    model.double()

    data_set_train_targets_cpu = data_set_train.targets[0:1000].cpu().numpy()
    data_set_test_targets_cpu = data_set_test.targets[0:500].cpu().numpy()
   
    accuracy_list_train = np.zeros(num_epoch//calc_interval + 1)
    accuracy_list_test  = np.zeros(num_epoch//calc_interval + 1)

    data_loader = DataLoader(data_set_train, batch_size=batch_size, shuffle=True)
    print(f'DATA SIZE: {data_set_train.data.size()}')

    for epoch in range(num_epoch):
        for batch in data_loader:
            data, target = batch['data'], batch['target']
            outputs = model.extract(data)
            outputs = model.forward(outputs)
            loss = criteria(outputs, target)
            
            optimizer.zero_grad()   # Zerowanie gradientów, aby git auniknąć akumulacji w kolejnych krokach
            loss.backward()         # Backpropagation: Obliczenie gradientów
            optimizer.step()        # Aktualizacja wag
        
        print(f"Epoch [{epoch+1}/{num_epoch}]")

        if epoch % calc_interval == 0:
            model.eval()

            outputs_train = model.extract(data_set_train.data[0:1000])
            outputs_train = model.forward(outputs_train)
            
            predicted_classes_train = torch.argmax(outputs_train, dim=1)
        
            predicted_classes_cpu_train = predicted_classes_train.cpu().numpy()
        
            accuracy = accuracy_score(predicted_classes_cpu_train, data_set_train_targets_cpu)
            accuracy_list_train[epoch//calc_interval] = accuracy
            #---
            outputs_test = model.extract(data_set_test.data[0:500])
            outputs_test = model.forward(outputs_test)
            
            predicted_classes_test = torch.argmax(outputs_test, dim=1)
        
            predicted_classes_test = predicted_classes_test.cpu().numpy()
        
            accuracy = accuracy_score(predicted_classes_test, data_set_test_targets_cpu)
            accuracy_list_test[epoch//calc_interval] = accuracy



        save_model(model.state_dict(), f'model_{data_name}.pth')

    plt.plot(range(len(accuracy_list_test)), accuracy_list_train, label="TRAIN")
    plt.plot(range(len(accuracy_list_test)), accuracy_list_test, label="TEST")
    plt.title(f"Accuracy Score for {data_name}")
    plt.legend()
    plt.savefig(f'accuracy_{data_name}.png')          
