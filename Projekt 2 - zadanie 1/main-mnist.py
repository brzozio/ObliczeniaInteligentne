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
from sklearn.metrics import confusion_matrix, accuracy_score
from voronoi import plot_decision_boundary, voronoi
from scipy.spatial import Voronoi


if __name__ == "__main__":
    train: bool           = False
    num_epochs            = 2000
    continue_train: bool = False
    print(torch.version.cuda)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'DEVICE RUNING: {device}')
    #data_set, features_size, class_size, data_name, hidden_neurons = datasets_get.mnist_flatten(device, train)
    #data_set, features_size, class_size, data_name, hidden_neurons = datasets_get.mnist_extr_PCA(device, train)
    #data_set, features_size, class_size, data_name, hidden_neurons = datasets_get.mnist_extr_TSNE(device, train, 'train' if train is True else 'test')
    data_set, features_size, class_size, data_name, hidden_neurons = datasets_get.mnist_extr_3(device, train, 'train' if train is True else 'test')
    #data_set, features_size, class_size, data_name, hidden_neurons = datasets_get.mnist_extr_4(device, train, 'train' if train is True else 'test')

    X_train, X_test, y_train, y_test = train_test_split(data_set.data, data_set.targets, test_size=0.2,  random_state=42)


    model = MLP(input_size=features_size, hidden_layer_size=hidden_neurons, classes=class_size) #input size to ilosc cech
    criteria = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    if continue_train is True:
        model.load_state_dict(load_model(f'model_{data_name}.pth'))
        
    model.to(device)

    if train is True:
        #Trenowanie modelu
        data_set.data    = X_train
        data_set.targets = y_train
        model.train()
        model.double()
        data_loader = DataLoader(data_set, batch_size=1024, shuffle=True) 

        for epoch in range(num_epochs):
            for batch in data_loader:
                data, target = batch['data'].to(device), batch['target'].to(device)
                outputs = model(batch['data'])
                loss = criteria(outputs, batch['target'])
                optimizer.zero_grad()   # Zerowanie gradientów, aby git auniknąć akumulacji w kolejnych krokach
                loss.backward()         # Backpropagation: Obliczenie gradientów
                optimizer.step()        # Aktualizacja wag
            print(f"Epoch [{epoch+1}/{num_epochs}]  Loss: {loss.item():.4f}   - {data_name}")
            if epoch % 200 == 0: 
                save_model(model.state_dict(), f'model_{data_name}.pth') #Zapisz model co 1_000 epok
                print(f"SAVED MODEL: model_{data_name}.pth at epoch [{epoch}]")

        save_model(model.state_dict(), f'model_{data_name}.pth') #Zapisz model na koniec trenignu - koniec epok
    else:
        #Klasyfikowanie danych
        model = MLP(input_size=features_size, hidden_layer_size=hidden_neurons, classes=class_size)
        model.load_state_dict(load_model(f'model_{data_name}.pth'))
        model.eval()
        model.double()
        model.to(device)
        data_set.data    = X_test 
        data_set.targets = y_test

        outputs = model(X_test)
        print(f"OUTPUS: {outputs}")
        softmax = torch.nn.Softmax(dim=1)
        probabilities = softmax(outputs)

        predicted_classes = torch.argmax(probabilities, dim=1)

        print(f'PREDICTED CLASSES: {predicted_classes}')
        print(f"ORIGINAL CLASSES: {data_set.targets}")

        plt.figure(figsize=(10, 7))        
        predicted_classes_cpu = predicted_classes.cpu().numpy()
        targets_cpu           = data_set.targets.cpu().numpy()
        sb.heatmap(confusion_matrix(targets_cpu,predicted_classes_cpu), annot=True, cmap='Blues', fmt='g')
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.show()
        accuracy = accuracy_score(predicted_classes_cpu, targets_cpu)
        print(f'ACCURACY SCORE FOR {data_name}: {accuracy:.4f}')

        #Diagram Voronoi'a oraz granice decyzyjne dla ekstrakcji do 2 cech
        if data_name is 'mnist_2_features_TSNE' or data_name is 'mnist_2_features_PCA': 
            model.to('cpu')
            #plot_decision_boundary(X=data_set.data.cpu(), func=lambda X: model(X), y_true=data_set.targets.cpu())
            #plot_decision_boundary(X=data_set.data.cpu(), func=lambda X: model(X))
            vor = Voronoi(data_set.data.cpu())
            voronoi(vor=vor, etykiety=predicted_classes_cpu)
