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
from model import CNN


def execute_model(data_set, model, batch_size, data_name, train: bool = False, continue_train: bool = False):
    num_epochs            = 10_000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'CUDA VERSION: {torch.version.cuda}')
    print(f'DEVICE RUNING: {device}')

    criteria = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    if continue_train is True:
        model.load_state_dict(load_model(f'model_{data_name}.pth'))
        
    model.to(device)

    if train is True:
        model.train()
        model.double()
        data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=True)

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
            
            if epoch % 10 == 0: 
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
