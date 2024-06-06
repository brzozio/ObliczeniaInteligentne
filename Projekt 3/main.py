"""
    1. MLP dla zbioru danych Iris
        Technika: Lime
        Opis: Lime pozwala na lokalne wyjaśnienia decyzji modelu, poprzez zastąpienie modelu nieliniowego modelem liniowym dla małych perturbacji danych wejściowych.
        Przykład Użycia: Użyj funkcji Lime z Captum, aby wyjaśnić decyzje modelu dla poszczególnych przykładów z zbioru danych Iris.
    
    2. MLP dla zbioru danych Wine
        Technika: Integrated Gradients
        Opis: Metoda Integrated Gradients polega na obliczeniu gradientów modelu w odniesieniu do wartości referencyjnych, co pomaga w zrozumieniu, które cechy najbardziej wpływają na decyzje modelu.
        Przykład Użycia: Użyj IntegratedGradients z Captum, aby wyjaśnić, które cechy win mają największy wpływ na klasyfikację.
    
    3. MLP dla zbioru danych Breast Cancer
        Technika: Saliency
        Opis: Saliency maps pomagają zidentyfikować, które wejścia są najbardziej krytyczne dla decyzji modelu poprzez obliczenie pierwszych pochodnych modelu.
        Przykład Użycia: Użyj Saliency z Captum, aby zobaczyć, które cechy wpływają na diagnozę raka piersi.
    
    4. Sieci dla zbioru danych MNIST
        MLP z ekstrakcją cech:
            Technika: Feature Ablation
            Opis: Feature Ablation polega na systematycznym usuwaniu każdej cechy i obserwacji wpływu na wynik modelu, co pomaga zrozumieć ważność poszczególnych cech.
            Przykład Użycia: Użyj FeatureAblation z Captum, aby ocenić wpływ poszczególnych cech na klasyfikację cyfr.
        CNN:
            Technika: Gradient Shap
            Opis: Gradient Shap łączy gradienty z metodą Shapley Values, co pozwala na bardziej precyzyjne wyjaśnienie wpływu cech na decyzje modelu.
            Przykład Użycia: Użyj GradientShap z Captum, aby wyjaśnić, które piksele w obrazach cyfr są najważniejsze dla klasyfikacji.
    
    5. CNN dla zbioru danych CIFAR10
        Technika: Guided GradCAM
        Opis: Guided GradCAM łączy GradCAM z Guided Backpropagation, co pozwala na tworzenie wysoce interpretowalnych map ciepła, które pokazują, które regiony obrazu są najważniejsze dla decyzji modelu.
        Przykład Użycia: Użyj GuidedGradCAM z Captum, aby zidentyfikować kluczowe obszary na obrazach CIFAR10 wpływające na klasyfikację.
"""
import datasets_get
from model_CNN import CNN_tanh
from model_MLP import MLP
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import torch
from captum.attr import visualization as viz
from captum.attr import IntegratedGradients
from captum.attr import Saliency
from captum.attr import LayerGradCam
from captum.attr import Lime
from captum.attr import GuidedBackprop
import seaborn as sb
from sklearn.metrics import confusion_matrix, accuracy_score
import torch.nn as nn
import torch.optim as optim


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_CNN_mnist         = CNN_tanh(in_side_len=28, in_channels=1, cnv0_out_channels=12, 
                                   cnv1_out_channels=16, lin0_out_size=100, lin1_out_size=10, 
                                   convolution_kernel=3, pooling_kernel=2, reduce_to_dim2=False)

model_CNN_cifar         = CNN_tanh(in_side_len=32, in_channels=3, cnv0_out_channels=15, 
                                   cnv1_out_channels=16, lin0_out_size=128, lin1_out_size=10, 
                                   convolution_kernel=7, pooling_kernel=2, reduce_to_dim2=False)

model_MLP_iris          = MLP(input_size=4, hidden_layer_size=3, classes=3)
model_MLP_wine          = MLP(input_size=13, hidden_layer_size=7, classes=3)
model_MLP_breast_cancer = MLP(input_size=30, hidden_layer_size=15, classes=2)

model_MLP_mnist_conv    = MLP(input_size=10, hidden_layer_size=40, classes=10)
model_MLP_mnist_diff    = MLP(input_size=10, hidden_layer_size=84, classes=10)

data_MLP_iris           = datasets_get.iris(device)
data_MLP_wine           = datasets_get.wine(device)
data_MLP_breast_cancer  = datasets_get.breast_cancer(device)
data_MLP_mnist_conv     = datasets_get.mnist_extr_conv(device, False, 'test')
data_MLP_mnist_diff     = datasets_get.mnist_extr_diff(device, False, 'test')
data_CNN_mnist          = datasets_get.mnist_to_cnn(device, True)
data_CNN_cifar          = datasets_get.cifar10_to_cnn(device, True)


def execute_model(data_set, model, data_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'CUDA VERSION: {torch.version.cuda}')
    print(f'DEVICE RUNING: {device}')


    model.load_state_dict(torch.load(f'./Projekt 3/models/{data_name}.pth'))
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


    plt.figure(figsize=(10, 7))        
    
    predicted_classes_cpu = predicted_classes.cpu().numpy()
    targets_cpu           = data_set.targets.cpu().numpy()
    
    sb.heatmap(confusion_matrix(targets_cpu,predicted_classes_cpu), annot=True, cmap='Blues', fmt='g')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title(f"Conf Matrix - {data_name}")
    plt.show()
    
    accuracy = accuracy_score(predicted_classes_cpu, targets_cpu)
    print(f'ACCURACY SCORE FOR {data_name}: {accuracy:.4f}')

        

def loading_state_dict():
    model_CNN_mnist.load_state_dict(torch.load('./Projekt 3/models/CNN_mnist.pth'))
    model_CNN_cifar.load_state_dict(torch.load('./Projekt 3/models/CNN_cifar.pth'))
    #model_MLP_iris.load_state_dict(torch.load('./Projekt 3/models/MLP_iris.pth'))
    model_MLP_wine.load_state_dict(torch.load('./Projekt 3/models/MLP_wine.pth'))
    model_MLP_breast_cancer.load_state_dict(torch.load('./Projekt 3/models/MLP_breast_cancer.pth'))
    model_MLP_mnist_conv.load_state_dict(torch.load('./Projekt 3/models/MLP_mnist_extr_conv.pth'))
    #model_MLP_mnist_diff.load_state_dict(torch.load('./Projekt 3/models/MLP_mnist_extr_diff.pth'))


def explain_saliency(model, data, target):
    model.double()  # Ensure the model is using double precision if required
    model.to(device)
    model.eval()
    
  

if __name__ == "__main__":
    loading_state_dict()
    #explain_gradCAM(model=model_CNN_cifar, data=data_CNN_cifar.data[1], target=data_CNN_cifar.targets[1], layer=model_CNN_cifar.conv0)
    #explain_gradCAM(model=model_CNN_mnist, data=data_CNN_mnist.data[1], target=data_CNN_mnist.targets[1], layer=model_CNN_mnist.conv0)

    '''
    DZIAŁA MODEL - ponizej funkcja do sprawdzania ewaluacji modelu

    execute_model(data_set=data_CNN_cifar, model=model_CNN_cifar, data_name='CNN_cifar')
    '''


    #explain_saliency(model=model_CNN_cifar, data=data_CNN_cifar.data, target=data_CNN_cifar.targets)
    explain_saliency(model=model_MLP_iris, data=data_MLP_iris.data, target=0)

    #explain_integrated_gradients(model=model_MLP_iris, data=data_MLP_iris.data[0], target=data_MLP_iris.targets[0])    