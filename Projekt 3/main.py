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
from model import CNN_tanh_compose as CNN_tanh
from model import MLP
import numpy as np 
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import torch
from captum.attr import visualization as viz
from captum.attr import IntegratedGradients, GuidedGradCam, Saliency, LayerGradCam, Lime, GuidedBackprop, FeatureAblation
import seaborn as sb
from sklearn.metrics import confusion_matrix, accuracy_score
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import os
import joblib

repo_name = "nteligentne"
path_script = os.path.dirname(os.path.realpath(__file__))
index = path_script.find(repo_name)
path_models = path_script + "\\models\\"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_CNN_mnist         = CNN_tanh(in_side_len=28, in_channels=1, cnv0_out_channels=12, 
                                   cnv1_out_channels=16, lin0_out_size=100, lin1_out_size=10, 
                                   convolution_kernel=3, pooling_kernel=2, reduce_to_dim2=False)

model_CNN_cifar         = CNN_tanh(in_side_len=32, in_channels=3, cnv0_out_channels=15, 
                                   cnv1_out_channels=16, lin0_out_size=128, lin1_out_size=10, 
                                   convolution_kernel=7, pooling_kernel=2, reduce_to_dim2=False)

model_MLP_iris          = MLP(input_size=4, hidden_layer_size=2, classes=3)
model_MLP_wine          = MLP(input_size=13, hidden_layer_size=7, classes=3)
model_MLP_breast_cancer = MLP(input_size=30, hidden_layer_size=15, classes=2)

model_MLP_mnist_conv    = MLP(input_size=10, hidden_layer_size=40, classes=10)
model_MLP_mnist_diff    = MLP(input_size=56, hidden_layer_size=84, classes=10)

data_MLP_iris           = datasets_get.iris(device)
data_MLP_wine           = datasets_get.wine(device)
data_MLP_breast_cancer  = datasets_get.breast_cancer(device)
data_MLP_mnist_conv     = datasets_get.mnist_extr_conv(device, False, 'test')
data_MLP_mnist_diff     = datasets_get.mnist_extr_diff(device, False, 'test')
data_CNN_mnist          = datasets_get.mnist_to_cnn(device, True)
data_CNN_cifar          = datasets_get.cifar10_to_cnn(device, True)

cifar10_classes = {
    0: "Airplane",
    1: "Automobile",
    2: "Bird",
    3: "Cat",
    4: "Deer",
    5: "Dog",
    6: "Frog",
    7: "Horse",
    8: "Ship",
    9: "Truck"
}

def execute_model(data_set, model, data_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'CUDA VERSION: {torch.version.cuda}')
    print(f'DEVICE RUNING: {device}')

    model.load_state_dict(torch.load(path_models + f'{data_name}.pth'))
    model.eval()
    model.double()
    model.to(device)
    if data_name == "CNN*":
        outputs = model.extract(data_set.data)
        outputs = model.forward(outputs)
        print(f"OUTPUS: {outputs}")
    else:
        outputs = model.forward(data_set.data)
        print(f"OUTPUS: {outputs}")

    
    softmax = torch.nn.Softmax(dim=1)
    probabilities = softmax(outputs)
    predicted_classes = torch.argmax(probabilities, dim=1)

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
    path_joblib = path_script + f"\\debug_temporaries\\{data_name}_pred_targets.joblib"
    joblib.dump(predicted_classes_cpu, path_joblib)

def testing_models_eval():
    execute_model(data_set=data_CNN_mnist, model=model_CNN_mnist, data_name='CNN_mnist')
    execute_model(data_set=data_CNN_cifar, model=model_CNN_cifar, data_name='CNN_cifar')
    execute_model(data_set=data_MLP_iris, model=model_MLP_iris, data_name='MLP_iris')
    execute_model(data_set=data_MLP_wine, model=model_MLP_wine, data_name='MLP_wine')
    execute_model(data_set=data_MLP_breast_cancer, model=model_MLP_breast_cancer, data_name='MLP_breast_cancer')
    execute_model(data_set=data_MLP_mnist_conv, model=model_MLP_mnist_conv, data_name='MLP_mnist_extr_conv')
    execute_model(data_set=data_MLP_mnist_diff, model=model_MLP_mnist_diff, data_name='MLP_mnist_extr_diff')

   
def loading_state_dict():
    model_CNN_mnist.load_state_dict(torch.load(path_models + 'CNN_mnist.pth'))
    model_CNN_cifar.load_state_dict(torch.load(path_models + 'CNN_cifar.pth'))
    model_MLP_iris.load_state_dict(torch.load(path_models + 'MLP_iris.pth'))
    model_MLP_wine.load_state_dict(torch.load(path_models + 'MLP_wine.pth'))
    model_MLP_breast_cancer.load_state_dict(torch.load(path_models + 'MLP_breast_cancer.pth'))
    model_MLP_mnist_conv.load_state_dict(torch.load(path_models + 'MLP_mnist_extr_conv.pth'))
    model_MLP_mnist_diff.load_state_dict(torch.load(path_models + 'MLP_mnist_extr_diff.pth'))

#Atrybucje
def get_attributions(model, input_tensor, target_class, method="saliency"):
    model.double()
    model.eval()
    model.to(device)

    input_tensor = input_tensor.requires_grad_(True)

    if method == "saliency":
        saliency = Saliency(model)
        attribution = saliency.attribute(input_tensor[0:1023], target=target_class[0:1023])
    elif method == "guided_gradcam":
        target_layer = model.conv1
        guided_gc = GuidedGradCam(model, target_layer)
        attribution = guided_gc.attribute(input_tensor[0:1023], target=target_class[0:1023])
    elif method == "lime":
        lime = Lime(model)
        attribution = lime.attribute(input_tensor[0:5], target=target_class[0:5])
    elif method == "feature_ablation":
        ftr_abl = FeatureAblation(model)
        attribution = ftr_abl.attribute(input_tensor[0:10], target=target_class[0:10])
    elif method == "integrated_gradients":
        integrated_gradients = IntegratedGradients(model)
        attribution = integrated_gradients.attribute(input_tensor[0:10], target=target_class[0:10])
    else:
        raise ValueError(f"Unknown method was specified: {method}")

    #print(f'ATTRIBUTION for {method} is: {attribution}, shape: {attribution.shape}, size: {attribution.dim}')
    return attribution

def tensor_to_attribution_heatmap(tensor):
    out = tensor.cpu().detach()
    for channel in range(1, out.size(0), 1):
        out[0] += out[channel]
    out = out[0] * 100
    return out



def visualize_attributions(attributions, input_tensor, model_name, method="saliency", target_tensor=None, example_datum=[0,1,2,3,4,5,6,7,8]):

    matplotlib.rcParams.update({'font.size': 7})

    if method == "saliency" or method == "feature_ablation" or method == "integrated_gradients":
        #WORKING
        plt.figure(figsize=(10, 5))
        sb.barplot(x=range(len(attributions[example_datum[0]])), y=attributions[example_datum[0]].cpu().detach().numpy())
        plt.xlabel('Feature Index')
        plt.ylabel('Attribution')
        plt.title(f'{method} for {model_name} - Target: [{target_tensor[0]}]')
        plt.show()
        
    elif method == "guided_gradcam_separate_ch":
        #WORKING
        pred_class = joblib.load(path_script + f"\\debug_temporaries\\{model_name.split()[0]}_{model_name.split()[1]}_pred_targets.joblib")
        _, ax = plt.subplots(3,4)

        ax[0,0].imshow(input_tensor[0].cpu().detach().numpy().transpose(1,2,0)/255.0)
        ax[0,1].imshow(attributions[0][0].cpu().detach().numpy(), cmap='Reds')
        ax[0,2].imshow(attributions[0][1].cpu().detach().numpy(), cmap='Greens')
        ax[0,3].imshow(attributions[0][2].cpu().detach().numpy(), cmap='Blues')
        ax[0,0].set_title(f"predicted class: {cifar10_classes[pred_class[0]]}")
        
        ax[1,0].imshow(input_tensor[1].cpu().detach().numpy().transpose(1,2,0)/255.0)
        ax[1,1].imshow(attributions[1][0].cpu().detach().numpy(), cmap='Reds')
        ax[1,2].imshow(attributions[1][1].cpu().detach().numpy(), cmap='Greens')
        ax[1,3].imshow(attributions[1][2].cpu().detach().numpy(), cmap='Blues')
        ax[1,0].set_title(f"predicted class: {cifar10_classes[pred_class[1]]}")
        
        ax[2,0].imshow(input_tensor[example_datum[2]].cpu().detach().numpy().transpose(1,2,0)/255.0)
        ax[2,1].imshow(attributions[example_datum[2]][0].cpu().detach().numpy(), cmap='Reds')
        ax[2,2].imshow(attributions[example_datum[2]][1].cpu().detach().numpy(), cmap='Greens')
        ax[2,3].imshow(attributions[example_datum[2]][2].cpu().detach().numpy(), cmap='Blues')

        for i in range(3):
            for j in range(4):
                ax[i,j].tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=False)
                ax[i,j].tick_params(axis='y',which='both',left=False,right=False,labelleft=False)

        plt.show()
        
    elif method == "guided_gradcam":
        #WORKING
        _, ax = plt.subplots(3,6, figsize=[5,8])
        pred_class = joblib.load(path_script + f"\\debug_temporaries\\{model_name.split()[0]}_{model_name.split()[1]}_pred_targets.joblib")
        if model_name.split()[1] == "Cifar":
            ax[0,0].set_title(f"predicted class: {cifar10_classes[pred_class[example_datum[0]]]}")
            ax[1,0].set_title(f"predicted class: {cifar10_classes[pred_class[example_datum[1]]]}")
            ax[2,0].set_title(f"predicted class: {cifar10_classes[pred_class[example_datum[2]]]}")
            ax[0,2].set_title(f"predicted class: {cifar10_classes[pred_class[example_datum[3]]]}")
            ax[1,2].set_title(f"predicted class: {cifar10_classes[pred_class[example_datum[4]]]}")
            ax[2,2].set_title(f"predicted class: {cifar10_classes[pred_class[example_datum[5]]]}")
            ax[0,4].set_title(f"predicted class: {cifar10_classes[pred_class[example_datum[6]]]}")
            ax[1,4].set_title(f"predicted class: {cifar10_classes[pred_class[example_datum[7]]]}")
            ax[2,4].set_title(f"predicted class: {cifar10_classes[pred_class[example_datum[8]]]}")
        else: 
            ax[0,0].set_title(f"predicted class: {pred_class[example_datum[0]]}")
            ax[1,0].set_title(f"predicted class: {pred_class[example_datum[1]]}")
            ax[2,0].set_title(f"predicted class: {pred_class[example_datum[2]]}")
            ax[0,2].set_title(f"predicted class: {pred_class[example_datum[3]]}")
            ax[1,2].set_title(f"predicted class: {pred_class[example_datum[4]]}")
            ax[2,2].set_title(f"predicted class: {pred_class[example_datum[5]]}")
            ax[0,4].set_title(f"predicted class: {pred_class[example_datum[6]]}")
            ax[1,4].set_title(f"predicted class: {pred_class[example_datum[7]]}")
            ax[2,4].set_title(f"predicted class: {pred_class[example_datum[8]]}")


        format_to_im = lambda tensor : \
            tensor.cpu().detach().numpy().transpose(1,2,0)/255
        
        ax[0,0].imshow(format_to_im(input_tensor[example_datum[0]]))
        ax[0,1].imshow(tensor_to_attribution_heatmap(attributions[example_datum[0]]), cmap='seismic', vmin=-1.0, vmax=1.0)

        ax[1,0].imshow(format_to_im(input_tensor[example_datum[1]]))
        ax[1,1].imshow(tensor_to_attribution_heatmap(attributions[example_datum[1]]), cmap='seismic', vmin=-1.0, vmax=1.0)
        
        ax[2,0].imshow(format_to_im(input_tensor[example_datum[2]]))        
        ax[2,1].imshow(tensor_to_attribution_heatmap(attributions[example_datum[2]]), cmap='seismic', vmin=-1.0, vmax=1.0)
        
        ax[0,2].imshow(format_to_im(input_tensor[example_datum[3]]))
        ax[0,3].imshow(tensor_to_attribution_heatmap(attributions[example_datum[3]]), cmap='seismic', vmin=-1.0, vmax=1.0)

        ax[1,2].imshow(format_to_im(input_tensor[example_datum[4]]))
        ax[1,3].imshow(tensor_to_attribution_heatmap(attributions[example_datum[4]]), cmap='seismic', vmin=-1.0, vmax=1.0)
        
        ax[2,2].imshow(format_to_im(input_tensor[example_datum[5]]))        
        ax[2,3].imshow(tensor_to_attribution_heatmap(attributions[example_datum[5]]), cmap='seismic', vmin=-1.0, vmax=1.0)
        
        ax[0,4].imshow(format_to_im(input_tensor[example_datum[6]]))
        ax[0,5].imshow(tensor_to_attribution_heatmap(attributions[example_datum[6]]), cmap='seismic', vmin=-1.0, vmax=1.0)

        ax[1,4].imshow(format_to_im(input_tensor[example_datum[7]]))
        ax[1,5].imshow(tensor_to_attribution_heatmap(attributions[example_datum[7]]), cmap='seismic', vmin=-1.0, vmax=1.0)
        
        ax[2,4].imshow(format_to_im(input_tensor[example_datum[8]]))        
        ax[2,5].imshow(tensor_to_attribution_heatmap(attributions[example_datum[8]]), cmap='seismic', vmin=-1.0, vmax=1.0)
        

        for i in range(3):
            for j in range(6):
                ax[i,j].tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=False)
                ax[i,j].tick_params(axis='y',which='both',left=False,right=False,labelleft=False)
        
        plt.suptitle(f"xAI for {model_name}, Method: {method}", fontname= 'Arial', fontsize = 20, fontweight = 'bold')
        plt.show()

    elif method == "lime":
        #WORKING
        _, ax = plt.subplots(3,1)
        ax[0].imshow(attributions[example_datum[0]][0].cpu().detach().numpy(), cmap='seismic')
        ax[1].imshow(attributions[example_datum[1]][0].cpu().detach().numpy(), cmap='seismic')
        ax[2].imshow(attributions[example_datum[2]][0].cpu().detach().numpy(), cmap='seismic')
        
        for i in range(3):
            ax[i].tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=False)
            ax[i].tick_params(axis='y',which='both',left=False,right=False,labelleft=False)
            
        plt.show()



if __name__ == "__main__":
    loading_state_dict()

    #Saliency Map oblicza gradienty wyniku modelu względem cech wejściowych, aby stworzyć mapę, która pokazuje, które cechy najbardziej wpływają na wynik modelu.
    #Guided Grad-CAM łączy Grad-CAM (Gradient-weighted Class Activation Mapping) z Guided Backpropagation, aby wygenerować wizualizację, która pokazuje, które części obrazu najbardziej wpływają na decyzję modelu.
    #Lime - Lime (Local Interpretable Model-agnostic Explanations) działa poprzez tworzenie prostego modelu liniowego w okolicy punktu, który chcemy wyjaśnić, aby zrozumieć, jak różne cechy wpływają na wynik modelu.
    #Integrated Gradients oblicza średnią gradientów modelu względem cech wejściowych na ścieżce od punktu początkowego (np. zerowego wektora) do rzeczywistego punktu wejściowego, aby uzyskać wyjaśnienie wpływu cech
    #Feature Ablation mierzy wpływ każdej cechy na wynik modelu poprzez sukcesywne usuwanie (ablacja) każdej cechy i obserwowanie zmiany w wyniku modelu.
 
    """
    #Saliency Map oblicza gradienty wyniku modelu względem cech wejściowych, aby stworzyć mapę, która pokazuje, które cechy najbardziej wpływają na wynik modelu.
    saliency_attributions = get_attributions(model=model_MLP_breast_cancer, input_tensor=data_MLP_breast_cancer.data, target_class=data_MLP_breast_cancer.targets, method="saliency")
    visualize_attributions(saliency_attributions, input_tensor=data_MLP_breast_cancer.data, model_name="MLP Breast Cancer", method="saliency", target_tensor=data_MLP_breast_cancer.targets)
  
    saliency_attributions = get_attributions(model=model_MLP_iris, input_tensor=data_MLP_iris.data, target_class=data_MLP_iris.targets, method="saliency")
    visualize_attributions(saliency_attributions, input_tensor=data_MLP_iris.data, model_name="MLP Iris", method="saliency", target_tensor=data_MLP_iris.targets)
  
    saliency_attributions = get_attributions(model=model_MLP_wine, input_tensor=data_MLP_wine.data, target_class=data_MLP_wine.targets, method="saliency")
    visualize_attributions(saliency_attributions, input_tensor=data_MLP_wine.data, model_name="MLP Wine", method="saliency", target_tensor=data_MLP_wine.targets)
    
    saliency_attributions = get_attributions(model=model_MLP_mnist_conv, input_tensor=data_MLP_mnist_conv.data, target_class=data_MLP_mnist_conv.targets, method="saliency")
    visualize_attributions(saliency_attributions, input_tensor=data_MLP_mnist_conv.data, model_name="MLP Mnist Conv", method="saliency", target_tensor=data_MLP_mnist_conv.targets)
    
    saliency_attributions = get_attributions(model=model_MLP_mnist_diff, input_tensor=data_MLP_mnist_diff.data, target_class=data_MLP_mnist_diff.targets, method="saliency")
    visualize_attributions(saliency_attributions, input_tensor=data_MLP_mnist_diff.data, model_name="MLP Mnist Diff", method="saliency", target_tensor=data_MLP_mnist_diff.targets)

    # Guided Grad-CAM łączy Grad-CAM (Gradient-weighted Class Activation Mapping) z Guided Backpropagation, aby wygenerować wizualizację, która pokazuje, które części obrazu najbardziej wpływają na decyzję modelu.

    """
    gradcam_attr = get_attributions(model=model_CNN_cifar, input_tensor=data_CNN_cifar.data, target_class=data_CNN_cifar.targets, method="guided_gradcam")
    visualize_attributions(gradcam_attr, input_tensor=data_CNN_cifar.data, model_name="CNN Cifar",  method="guided_gradcam", example_datum=[5,8,13,67,15,17,32,45,23])
    
    gradcam_attr = get_attributions(model=model_CNN_mnist, input_tensor=data_CNN_mnist.data, target_class=data_CNN_mnist.targets, method="guided_gradcam")
    visualize_attributions(gradcam_attr, input_tensor=data_CNN_mnist.data, model_name="CNN Mnist",  method="guided_gradcam", example_datum=[5,8,13,67,15,17,32,45,23])
  
    #Lime - Lime (Local Interpretable Model-agnostic Explanations) działa poprzez tworzenie prostego modelu liniowego w okolicy punktu, który chcemy wyjaśnić, aby zrozumieć, jak różne cechy wpływają na wynik modelu.
    #lime_attr = get_attributions(model=model_MLP_mnist_diff, input_tensor=data_MLP_mnist_diff.data, target_class=data_MLP_mnist_diff.targets, method="lime")
    #visualize_attributions(lime_attr, input_tensor=data_MLP_mnist_diff.data, model_name="MLP Mnist Diff",  method="lime")
    
    #lime_attr = get_attributions(model=model_MLP_mnist_conv, input_tensor=data_MLP_mnist_conv.data, target_class=data_MLP_mnist_conv.targets, method="lime")
    #visualize_attributions(lime_attr, input_tensor=data_MLP_mnist_conv.data, model_name="MLP Mnist Conv",  method="lime")
    
    
    #Integrated Gradients oblicza średnią gradientów modelu względem cech wejściowych na ścieżce od punktu początkowego (np. zerowego wektora) do rzeczywistego punktu wejściowego, aby uzyskać wyjaśnienie wpływu cech
    #intrgrad_attr = get_attributions(model=model_MLP_wine, input_tensor=data_MLP_wine.data, target_class=data_MLP_wine.targets, method="integrated_gradients")
    #visualize_attributions(intrgrad_attr, input_tensor=data_MLP_wine.data, model_name="MLP Wine", method="integrated_gradients", target_tensor=data_MLP_wine.targets)

    #Feature Ablation mierzy wpływ każdej cechy na wynik modelu poprzez sukcesywne usuwanie (ablacja) każdej cechy i obserwowanie zmiany w wyniku modelu.
    #featurueabl_attr = get_attributions(model=model_MLP_mnist_diff, input_tensor=data_MLP_mnist_diff.data, target_class=data_MLP_mnist_diff.targets, method="feature_ablation")
    #visualize_attributions(featurueabl_attr, input_tensor=data_MLP_mnist_diff.data, model_name="MLP Mnist Diff", method="feature_ablation", target_tensor=data_MLP_mnist_diff.targets)

    
