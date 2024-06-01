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
from captum.attr import FeatureAblation
from captum.attr import visualization as viz
from captum.attr import IntegratedGradients
from captum.attr import Saliency
from captum.attr import LayerGradCam
from captum.attr import Lime


model_CNN_mnist         = CNN_tanh(in_side_len=28, in_channels=1, cnv0_out_channels=12, cnv1_out_channels=16, lin0_out_size=100, lin1_out_size=10, convolution_kernel=3, pooling_kernel=2, reduce_to_dim2=False)
model_CNN_cifar         = CNN_tanh(in_side_len=32, in_channels=3, cnv0_out_channels=15, cnv1_out_channels=16, lin0_out_size=128, lin1_out_size=10, convolution_kernel=7, pooling_kernel=2, reduce_to_dim2=False)

model_MLP_iris          = MLP(input_size=4, hidden_layer_size=3, classes=3)
model_MLP_wine          = MLP(input_size=13, hidden_layer_size=7, classes=3)
model_MLP_breast_cancer = MLP(input_size=30, hidden_layer_size=15, classes=2)

model_MLP_mnist_conv         = MLP(input_size=10, hidden_layer_size=40, classes=10)
model_MLP_mnist_diff         = MLP(input_size=10, hidden_layer_size=84, classes=10)

data_MLP_iris  = datasets_get.iris()
data_MLP_wine  = datasets_get.wine()
data_MLP_breast_cancer  = datasets_get.breast_cancer()
data_MLP_mnist_conv = datasets_get.mnist_extr_conv()
data_MLP_mnist_diff = datasets_get.mnist_extr_diff()

data_CNN_mnist = datasets_get.mnist_to_cnn()
data_CNN_cifar = datasets_get.cifar10_to_cnn()


def loading_state_dict():
    model_CNN_mnist.load_state_dict('./models/CNN_mnist.pth')
    model_CNN_cifar.load_state_dict('./models/CNN_cifar.pth')
    model_MLP_iris.load_state_dict('./models/MLP_iris.pth')
    model_MLP_wine.load_state_dict('./models/MLP_wine.pth')
    model_MLP_breast_cancer.load_state_dict('./models/MLP_breast_cancer.pth')
    model_MLP_mnist_conv.load_state_dict('./models/MLP_mnist_extr_conv.pth')
    model_MLP_mnist_diff.load_state_dict('./models/MLP_mnist_extr_diff.pth')

def explain_lime(model, data, target, feature_names):
    model.eval()
    lime = Lime(model)
    attr = lime.attribute(data, target=target)
    attr = attr.detach().numpy()

    viz.visualize_text([attr], feature_names, title="LIME Attribution")

def explain_gradCAM(model, data, target, layer):
    model.eval()
    gradcam = LayerGradCam(model, layer)
    attr = gradcam.attribute(data, target=target)
    attr = attr.detach().numpy()

    viz.visualize_image_attr(attr, data.permute(1, 2, 0).numpy(), method="heat_map", title="GradCAM Attribution")

def explain_saliency(model, data, target):
    model.eval()
    saliency = Saliency(model)
    attr = saliency.attribute(data, target=target, abs=True)
    attr = attr.detach().numpy()

    viz.visualize_image_attr(attr, data.permute(1, 2, 0).numpy(), method="heat_map", title="Saliency Attribution")

def explain_integrated_gradients(model, data, target, baseline=None):
    model.eval()
    integrated_gradients = IntegratedGradients(model)
    attr = integrated_gradients.attribute(data, baseline, target=target, n_steps=50)
    attr = attr.detach().numpy()

    viz.visualize_image_attr(attr, data.permute(1, 2, 0).numpy(), method="heat_map", title="Integrated Gradients Attribution")


if __name__ == "__main__":
    loading_state_dict()

    