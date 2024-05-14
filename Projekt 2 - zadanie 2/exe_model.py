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
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, silhouette_score
from voronoi import plot_decision_boundary, voronoi
from scipy.spatial import Voronoi
import torch.nn as nn
from model import CNN


def execute_model(model, data_name, model_path, train: bool = False, continue_train: bool = False):
    num_epochs            = 10_000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'CUDA VERSION: {torch.version.cuda}')
    print(f'DEVICE RUNING: {device}')

    data_set = datasets_get.mnist_to_cnn(device,  train)

    criteria = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    if continue_train is True:
        model.load_state_dict(load_model(model_path))
        
    model.to(device)

    if train is True:
        model.train()
        model.double()
        data_loader = DataLoader(data_set, batch_size=8000, shuffle=True)

        for epoch in range(num_epochs):
            for batch in data_loader:
                data, target = batch['data'].to(device), batch['target'].to(device)
                outputs = model.forward(data)
                loss = criteria(outputs, target)
                
                optimizer.zero_grad()   # Zerowanie gradientów, aby git auniknąć akumulacji w kolejnych krokach
                loss.backward()         # Backpropagation: Obliczenie gradientów
                optimizer.step()        # Aktualizacja wag
            
            print(f"Epoch [{epoch+1}/{num_epochs}]  Loss: {loss.item():.5f}   - {data_name}")
            
            if epoch % 10 == 0: 
                save_model(model.state_dict(), model_path) #Zapisz model co 1_000 epok
                print(f"SAVED MODEL: model_{data_name}.pth at epoch [{epoch}]")

        save_model(model.state_dict(), model_path) #Zapisz model na koniec trenignu - koniec epok
    else:

        model.load_state_dict(load_model(model_path))
        model.eval()
        model.double()
        model.to(device)

        outputs = model(data_set.data)
        print(f"OUTPUS: {outputs}")
        
        softmax = torch.nn.Softmax(dim=1)
        probabilities = softmax(outputs)
        predicted_classes = torch.argmax(probabilities, dim=1)

        print(f'PREDICTED CLASSES: {predicted_classes}')
        print(f"ORIGINAL CLASSES: {data_set.targets}")

        plt.figure(figsize=(10, 7))        
       
        predicted_classes_cpu = predicted_classes.cpu().numpy()
        targets_cpu           = data_set.targets.cpu().numpy()
        dataset_cpu           = data_set.data.cpu().numpy()
        
        sb.heatmap(confusion_matrix(targets_cpu,predicted_classes_cpu), annot=True, cmap='Blues', fmt='g')
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.show()
        
        accuracy = accuracy_score(predicted_classes_cpu, targets_cpu)
        print(f'ACCURACY SCORE FOR {data_name}: {accuracy:.4f}')

        #Diagram Voronoi'a oraz granice decyzyjne dla ekstrakcji do 2 cech
        if model.reduce_to_dim2:

            data = model.extract(data_set.data)
            # print(data.size())
            data = data.view(-1,2)
            model.to('cpu')

            #plot_decision_boundary(X=data_set.data.cpu(), func=lambda X: model(X), y_true=data_set.targets.cpu())

            plot_decision_boundary(X=data.cpu(), func=lambda X: model.forward(X))
            
            vor = Voronoi(data_set.data.cpu())
            voronoi(vor=vor, etykiety=predicted_classes_cpu)
