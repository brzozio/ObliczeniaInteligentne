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


if __name__ == "__main__":
    train: bool   = True
    num_epochs    = 1_000_000
    continue_train: bool = True
    print(torch.version.cuda)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    data_set, features_size, class_size, data_name, hidden_neurons = datasets_get.iris(device)
    #data_set, features_size, class_size, data_name, hidden_neurons = datasets_get.wine(device)
    #data_set, features_size, class_size, data_name, hidden_neurons = datasets_get.breast_cancer(device)

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
        for epoch in range(num_epochs):
            outputs = model(data_set.data)
            loss = criteria(outputs, data_set.targets)
            optimizer.zero_grad()   # Zerowanie gradientów, aby uniknąć akumulacji w kolejnych krokach
            loss.backward()         # Backpropagation: Obliczenie gradientów
            optimizer.step()        # Aktualizacja wag
            print(f"Epoch [{epoch+1}/{num_epochs}]  Loss: {loss.item():.8f}   - {data_name}")
            if epoch % 2000 == 0: 
                save_model(model.state_dict(), f'model_{data_name}.pth') #Zapisz model co 1_000 epok
                print(f"SAVED MODEL: model_{data_name}.pth at epoch [{epoch}]")
        print(f"Epoch [{epoch+1}/{num_epochs}]  Loss: {loss.item():.4f}   - {data_name}")

        save_model(model.state_dict(), f'model_{data_name}.pth')
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
        print(f"PROBS: {probabilities}")
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


