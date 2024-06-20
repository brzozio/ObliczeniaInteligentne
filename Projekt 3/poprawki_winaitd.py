import datasets_get
from model import CNN_tanh_compose as CNN_tanh
from model import MLP
import numpy as np 
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import torch
from captum.attr import visualization as viz
from captum.attr import IntegratedGradients, GuidedGradCam, Saliency, LayerGradCam, Lime, GuidedBackprop, FeatureAblation, ShapleyValueSampling, DeepLiftShap
import seaborn as sb
from sklearn.metrics import confusion_matrix, accuracy_score
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import os
import joblib
from matplotlib.gridspec import GridSpec

repo_name = "nteligentne"
path_script = os.path.dirname(os.path.realpath(__file__))
index = path_script.find(repo_name)
path_models = path_script + "\\models\\"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_MLP_iris          = MLP(input_size=4, hidden_layer_size=2, classes=3)
model_MLP_wine          = MLP(input_size=13, hidden_layer_size=7, classes=3)
model_MLP_breast_cancer = MLP(input_size=30, hidden_layer_size=15, classes=2)

data_MLP_iris           = datasets_get.iris(device)
data_MLP_wine           = datasets_get.wine(device)
data_MLP_breast_cancer  = datasets_get.breast_cancer(device)


iris_classes = [
    "Iris-setosa",
    "Iris-versicolor",
    "Iris-virginica"
]

breast_cancer_classes = [
    "Benign",
    "Malignant"
]
wine_classes = [
    "Wine Class 0",
    "Wine Class 1",
    "Wine Class 2"
]
iris_features = [
    'sepal length (cm)',
    'sepal width (cm)',
    'petal length (cm)',
    'petal width (cm)']
wine_features = [
    'alcohol', 
    'malic_acid', 
    'ash', 
    'alcalinity_of_ash', 
    'magnesium', 
    'total_phenols', 
    'flavanoids', 
    'nonflavanoid_phenols', 
    'proanthocyanins', 
    'color_intensity', 
    'hue', 
    'od280/od315_of_diluted_wines', 
    'proline'
]
breast_cancer_features = [
    'mean radius', 
    'mean texture', 
    'mean perimeter', 
    'mean area', 
    'mean smoothness', 
    'mean compactness', 
    'mean concavity', 
    'mean concave points', 
    'mean symmetry', 
    'mean fractal dimension', 
    'radius error', 
    'texture error', 
    'perimeter error', 
    'area error', 
    'smoothness error', 
    'compactness error', 
    'concavity error', 
    'concave points error', 
    'symmetry error', 
    'fractal dimension error', 
    'worst radius', 
    'worst texture', 
    'worst perimeter', 
    'worst area', 
    'worst smoothness', 
    'worst compactness', 
    'worst concavity', 
    'worst concave points', 
    'worst symmetry', 
    'worst fractal dimension'
]

   
def loading_state_dict():
    model_MLP_iris.load_state_dict(torch.load(path_models + 'MLP_iris.pth'))
    model_MLP_wine.load_state_dict(torch.load(path_models + 'MLP_wine.pth'))
    model_MLP_breast_cancer.load_state_dict(torch.load(path_models + 'MLP_breast_cancer.pth'))


#Atrybucje
def get_attributions(model, input_tensor, target_class, method="saliency"):
    model.double()
    model.eval()
    model.to(device)
    input_tensor = input_tensor.requires_grad_(True)
    if method == "saliency":
        saliency = Saliency(model)
        attribution = saliency.attribute(input_tensor, target=target_class, abs=False)
    elif method == "feature_ablation":
        ftr_abl = FeatureAblation(model)
        attribution = ftr_abl.attribute(input_tensor, target=target_class)
    elif method == "integrated_gradients":
        integrated_gradients = IntegratedGradients(model)
        attribution = integrated_gradients.attribute(input_tensor, target=target_class)
    else:
        raise ValueError(f"Unknown method was specified: {method}")

    return attribution, input_tensor, target_class, 

def tensor_to_attribution_heatmap(tensor):
    out = tensor.cpu().detach()
    for channel in range(1, out.size(0), 1):
        out[0] += out[channel]
    out = out[0]
    return out

def visualize_attributions(attributions, input_tensor, model_name, method=None, target_tensor=None, features=None, target=None, classes=None):
    matplotlib.rcParams.update({'font.size': 7})
    
    if method == "saliency_barplot" or method == "integrated_gradients_barplot":
        _, ax = plt.subplots(1, 1, figsize=(10, 5))
        data = []

        for i, feature in enumerate(features):
            attr_np = attributions[:, i].cpu().detach().numpy()
            data.extend([(feature, attr) for attr in attr_np])

        df = pd.DataFrame(data, columns=["Feature", "Attribution"])

        sb.violinplot(x="Feature", y="Attribution", data=df, ax=ax, scale="width")
        ax.set_ylabel('Attribution value')
        ax.set_ylim()
        ax.set_title(f'{features}', fontsize=12, fontweight='bold')

        plt.suptitle(f"Class: {classes[target]}", fontname='Arial', fontsize=30, fontweight='bold')
        plt.tight_layout()
        plt.show()


def explain_MLP(model=None, data_set=None, model_name=None, target=None, features=None, classes=None):

    indices = []

    for idx, label in enumerate(data_MLP_iris.targets):
        if label == target:
            indices.append(idx)
    data_tensor = data_set.data[indices]
    targets = data_set.targets[indices]
    
    #saliency_attr, input_tensor, target_tensor = get_attributions(model=model, input_tensor=data_tensor, target_class=targets, method="saliency")
    #visualize_attributions(saliency_attr, input_tensor=input_tensor, model_name=model_name,  method="saliency_barplot", target_tensor=target_tensor, features=features, target=target)

    int_grd, input_tensor, target_tensor  = get_attributions(model=model, input_tensor=data_tensor, target_class=targets, method="integrated_gradients")
    visualize_attributions(int_grd, input_tensor, model_name=model_name,  method="integrated_gradients_barplot", target_tensor=target_tensor, features=features, target=target, classes=classes)
    

if __name__ == "__main__":
    
    loading_state_dict()
    explain_MLP(model=model_MLP_iris, data_set=data_MLP_iris, model_name="MLP Iris", target=0, features=iris_features, classes=iris_classes)
    explain_MLP(model=model_MLP_iris, data_set=data_MLP_iris, model_name="MLP Iris", target=1, features=iris_features, classes=iris_classes)
    explain_MLP(model=model_MLP_iris, data_set=data_MLP_iris, model_name="MLP Iris", target=2, features=iris_features, classes=iris_classes)

    explain_MLP(model=model_MLP_wine, data_set=data_MLP_wine, model_name="MLP Wine", target=0, features=wine_features, classes=wine_classes)
    explain_MLP(model=model_MLP_wine, data_set=data_MLP_wine, model_name="MLP Wine", target=1, features=wine_features, classes=wine_classes)
    explain_MLP(model=model_MLP_wine, data_set=data_MLP_wine, model_name="MLP Wine", target=2, features=wine_features, classes=wine_classes)
    
    explain_MLP(model=model_MLP_breast_cancer, data_set=data_MLP_breast_cancer, model_name="MLP Breast", target=0, features=breast_cancer_features, classes=breast_cancer_classes)
    explain_MLP(model=model_MLP_breast_cancer, data_set=data_MLP_breast_cancer, model_name="MLP Breast", target=1, features=breast_cancer_features, classes=breast_cancer_classes)

 
  