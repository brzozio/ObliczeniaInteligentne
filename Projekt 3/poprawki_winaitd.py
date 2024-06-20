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

iris_classes = [    "Iris-setosa",    "Iris-versicolor",    "Iris-virginica"]

breast_cancer_classes = [    "Benign",    "Malignant"
]
wine_classes = [    "Wine Class 0",    "Wine Class 1",    "Wine Class 2"]
iris_features = [    'sepal length (cm)',    'sepal width (cm)',    'petal length (cm)',    'petal width (cm)']
wine_features = [    'alcohol',     'malic_acid',     'ash',     'alcalinity_of_ash',     'magnesium',     'total_phenols',     'flavanoids',     'nonflavanoid_phenols',     'proanthocyanins',     'color_intensity',     'hue',     'od280/od315_of_diluted_wines',     'proline']
breast_cancer_features = ['mean radius',     'mean texture',     'mean perimeter',     'mean area',     'mean smoothness',     'mean compactness',     'mean concavity',     'mean concave points',     'mean symmetry',     'mean fractal dimension',     'radius error',     'texture error',     'perimeter error',     'area error',     'smoothness error',     'compactness error',     'concavity error',     'concave points error',     'symmetry error',     'fractal dimension error',     'worst radius',     'worst texture',     'worst perimeter',     'worst area',     'worst smoothness',     'worst compactness',     'worst concavity',     'worst concave points',     'worst symmetry',     'worst fractal dimension']
   
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

def visualize_attributions(attributions, input_tensor, model_name, method=None, target_tensor=None, features=None, target=None, classes=None, ax=None):
    matplotlib.rcParams.update({'font.size': 7})
    
    if method == "saliency_barplot" or method == "integrated_gradients_barplot":
        data = []

        for i, feature in enumerate(features):
            attr_np = attributions[:, i].cpu().detach().numpy()
            data.extend([(feature, attr) for attr in attr_np])

        df = pd.DataFrame(data, columns=["Feature", "Attribution"])

        sb.violinplot(x="Feature", y="Attribution", data=df, ax=ax, scale="width", palette=sb.color_palette("husl", len(features)))
        ax.set_ylabel('Attribution value')

        ax.set_title(f"Class: {classes[target]}", fontname='Arial', fontsize=16, fontweight='bold')
        
        plt.tight_layout()

def explain_MLP(model=None, data_set=None, model_name=None, features=None, classes=None, method=None):
    _, ax = plt.subplots(len(classes), 1, figsize=(10, 5))

    for target in range(len(classes)):
        indices = []

        for idx, label in enumerate(data_set.targets):
            if label == target:
                indices.append(idx)

        data_tensor = data_set.data[indices]
        targets = data_set.targets[indices]
        
        int_grd, input_tensor, target_tensor  = get_attributions(model=model, input_tensor=data_tensor, target_class=targets, method=method)
        visualize_attributions(int_grd, input_tensor, model_name=model_name,  method=f"{method}_barplot", target_tensor=target_tensor, features=features, target=target, classes=classes, ax=ax[target])
        if target < len(classes) - 1: 
            ax[target].tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=False)
            ax[target].set_xlabel('')

        if len(features) != 4: ax[target].tick_params(axis='x', rotation=90)
        attr, _, _ = get_attributions(model=model, input_tensor=data_set.data, target_class=data_set.targets, method=method)
        y_min = np.min(attr.cpu().detach().numpy()) - 5
        y_max = np.max(attr.cpu().detach().numpy()) + 5
        ax[target].set_ylim([y_min, y_max])

    plt.suptitle(f"{model_name.split()[1]} Dataset - {method} Attribution Distribution", fontsize=20, fontname="Arial", fontweight="bold")
    plt.show()

if __name__ == "__main__":
    
    loading_state_dict()
    explain_MLP(model=model_MLP_iris, data_set=data_MLP_iris, model_name="MLP Iris", features=iris_features, classes=iris_classes, method="integrated_gradients")
    explain_MLP(model=model_MLP_wine, data_set=data_MLP_wine, model_name="MLP Wine", features=wine_features, classes=wine_classes, method="integrated_gradients")
    explain_MLP(model=model_MLP_breast_cancer, data_set=data_MLP_breast_cancer, model_name="MLP Breast", features=breast_cancer_features, classes=breast_cancer_classes, method="integrated_gradients")