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
from matplotlib.gridspec import GridSpec

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
iris_classes = {
    0: "Iris-setosa",
    1: "Iris-versicolor",
    2: "Iris-virginica"
}
breast_cancer_classes = {
    0: "Benign",
    1: "Malignant"
}
wine_classes = {
    0: "Wine Class 0",
    1: "Wine Class 1",
    2: "Wine Class 2"
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
        attribution = lime.attribute(input_tensor[0:100], target=target_class[0:100])
    elif method == "feature_ablation":
        ftr_abl = FeatureAblation(model)
        attribution = ftr_abl.attribute(input_tensor[0:100], target=target_class[0:100])
    elif method == "integrated_gradients":
        integrated_gradients = IntegratedGradients(model)
        attribution = integrated_gradients.attribute(input_tensor[0:100], target=target_class[0:100])
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



def visualize_attributions(attributions, input_tensor, model_name, method=None, target_tensor=None, example_datum=[0,1,2,3,4,5,6,7,8]):

    matplotlib.rcParams.update({'font.size': 7})

    if method == "saliency_barplot" or method == "integrated_gradients_barplot": #Bar-plot
        #WORKING
        pred_class = joblib.load(path_script + f"\\debug_temporaries\\{model_name.split()[0]}_{model_name.split()[1]}_pred_targets.joblib")
        if model_name.split()[1] == "iris":
            _, ax = plt.subplots(3, 1, figsize=(10, 5))
            sb.barplot(x=range(len(attributions[example_datum[0]])), y=attributions[example_datum[0]].cpu().detach().numpy(), ax=ax[0])
            sb.barplot(x=range(len(attributions[example_datum[1]])), y=attributions[example_datum[1]].cpu().detach().numpy(), ax=ax[1])
            sb.barplot(x=range(len(attributions[example_datum[2]])), y=attributions[example_datum[2]].cpu().detach().numpy(), ax=ax[2])
            ax[0].set_title(f'{method} for {model_name} - Predicted: [{iris_classes[pred_class[example_datum[0]]]}]')
            ax[1].set_title(f'{method} for {model_name} - Predicted: [{iris_classes[pred_class[example_datum[1]]]}]')
            ax[2].set_title(f'{method} for {model_name} - Predicted: [{iris_classes[pred_class[example_datum[2]]]}]')
        elif model_name.split()[1] == "wine":
            _, ax = plt.subplots(3, 1, figsize=(10, 5))
            sb.barplot(x=range(len(attributions[example_datum[0]])), y=attributions[example_datum[0]].cpu().detach().numpy(), ax=ax[0])
            sb.barplot(x=range(len(attributions[example_datum[1]])), y=attributions[example_datum[1]].cpu().detach().numpy(), ax=ax[1])
            sb.barplot(x=range(len(attributions[example_datum[2]])), y=attributions[example_datum[2]].cpu().detach().numpy(), ax=ax[2])
            ax[0].set_title(f'{method} for {model_name} - Predicted: [{wine_classes[pred_class[example_datum[0]]]}]')
            ax[1].set_title(f'{method} for {model_name} - Predicted: [{wine_classes[pred_class[example_datum[1]]]}]')
            ax[2].set_title(f'{method} for {model_name} - Predicted: [{wine_classes[pred_class[example_datum[2]]]}]')
        elif model_name.split()[1] == "breast":
            _, ax = plt.subplots(2, 1, figsize=(10, 5))
            sb.barplot(x=range(len(attributions[example_datum[0]])), y=attributions[example_datum[0]].cpu().detach().numpy(), ax=ax[0])
            sb.barplot(x=range(len(attributions[example_datum[1]])), y=attributions[example_datum[1]].cpu().detach().numpy(), ax=ax[1])
            ax[0].set_title(f'{method} for {model_name} - Predicted: [{breast_cancer_classes[pred_class[example_datum[0]]]}]')
            ax[1].set_title(f'{method} for {model_name} - Predicted: [{breast_cancer_classes[pred_class[example_datum[1]]]}]')
        plt.xlabel('Feature Index')
        plt.ylabel('Attribution')
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
        
    elif method == "guided_gradcam" or method == "saliency_2" or method == "feature_ablation" or method == "lime":
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

    elif method == "differential":
        
        # for image in range(ilość obrazów)
        fig = plt.figure(figsize=(1.2, 2.38))
        gs = GridSpec(4, 4, figure=fig) 
        # image
        ax1 = fig.add_subplot(gs[0:3, 0:3])
        ax1.imshow(input_tensor[example_datum[0]][0].cpu().detach().numpy())
        # vertical axis - 28:56
        ax2 = fig.add_subplot(gs[0:3, 3])
        ax2.plot(attributions[example_datum[0]][28:56].cpu().detach().numpy(), range(28))
        # horizontal axis = 0:28
        ax3 = fig.add_subplot(gs[3, 0:3])
        ax3.plot(attributions[example_datum[0]][0:28].cpu().detach().numpy())
        ax3.invert_yaxis()

        ax1.tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=False)
        ax1.tick_params(axis='y',which='both',left=False,right=False,labelleft=False)
        ax2.tick_params(axis='y',which='both',left=False,right=True,labelleft=False)
        ax3.tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=False)
        ax3.yaxis.tick_right()

        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        plt.show()
        # zapisz subplot



def explain_CNN():
    lime_attr = get_attributions(model=model_CNN_cifar, input_tensor=data_CNN_cifar.data, target_class=data_CNN_cifar.targets, method="lime")
    visualize_attributions(lime_attr, input_tensor=data_CNN_cifar.data, model_name="CNN Cifar",  method="lime", example_datum=[5,8,13,67,15,17,32,45,23])

    lime_attr = get_attributions(model=model_CNN_mnist, input_tensor=data_CNN_mnist.data, target_class=data_CNN_mnist.targets, method="lime")
    visualize_attributions(lime_attr, input_tensor=data_CNN_mnist.data, model_name="CNN Mnist",  method="lime", example_datum=[5,8,13,67,15,17,32,45,23])

    ablation = get_attributions(model=model_CNN_cifar, input_tensor=data_CNN_cifar.data, target_class=data_CNN_cifar.targets, method="feature_ablation")
    visualize_attributions(ablation, input_tensor=data_CNN_cifar.data, model_name="CNN Cifar",  method="feature_ablation", example_datum=[5,8,13,67,15,17,32,45,23])

    gradcam_attr = get_attributions(model=model_CNN_mnist, input_tensor=data_CNN_mnist.data, target_class=data_CNN_mnist.targets, method="feature_ablation")
    visualize_attributions(gradcam_attr, input_tensor=data_CNN_mnist.data, model_name="CNN Mnist",  method="feature_ablation", example_datum=[5,8,13,67,15,17,32,45,23])
  
    
    gradcam_attr = get_attributions(model=model_CNN_cifar, input_tensor=data_CNN_cifar.data, target_class=data_CNN_cifar.targets, method="guided_gradcam")
    visualize_attributions(gradcam_attr, input_tensor=data_CNN_cifar.data, model_name="CNN Cifar",  method="guided_gradcam", example_datum=[5,8,13,67,15,17,32,45,23])
    
    gradcam_attr = get_attributions(model=model_CNN_mnist, input_tensor=data_CNN_mnist.data, target_class=data_CNN_mnist.targets, method="guided_gradcam")
    visualize_attributions(gradcam_attr, input_tensor=data_CNN_mnist.data, model_name="CNN Mnist",  method="guided_gradcam", example_datum=[5,8,13,67,15,17,32,45,23])


def explain_MLP():
    saliency_attr = get_attributions(model=model_MLP_iris, input_tensor=data_MLP_iris.data, target_class=data_MLP_iris.targets, method="saliency")
    visualize_attributions(saliency_attr, input_tensor=data_MLP_iris.data, model_name="MLP iris",  method="saliency_barplot", example_datum=[0,24,90])
    saliency_attr = get_attributions(model=model_MLP_wine, input_tensor=data_MLP_wine.data, target_class=data_MLP_wine.targets, method="saliency")
    visualize_attributions(saliency_attr, input_tensor=data_MLP_wine.data, model_name="MLP wine",  method="saliency_barplot", example_datum=[0,3,4])
    saliency_attr = get_attributions(model=model_MLP_breast_cancer, input_tensor=data_MLP_breast_cancer.data, target_class=data_MLP_breast_cancer.targets, method="saliency")
    visualize_attributions(saliency_attr, input_tensor=data_MLP_breast_cancer.data, model_name="MLP breast cancer",  method="saliency_barplot", example_datum=[5,8,13,67,15,17,32,45,23])



if __name__ == "__main__":
    """
    Saliency Map oblicza gradienty wyniku modelu względem cech wejściowych, aby stworzyć mapę, która pokazuje, które cechy najbardziej wpływają na wynik modelu.
    Guided Grad-CAM łączy Grad-CAM (Gradient-weighted Class Activation Mapping) z Guided Backpropagation, aby wygenerować wizualizację, która pokazuje, które części obrazu najbardziej wpływają na decyzję modelu.
    Lime - Lime (Local Interpretable Model-agnostic Explanations) działa poprzez tworzenie prostego modelu liniowego w okolicy punktu, który chcemy wyjaśnić, aby zrozumieć, jak różne cechy wpływają na wynik modelu.
    Integrated Gradients oblicza średnią gradientów modelu względem cech wejściowych na ścieżce od punktu początkowego (np. zerowego wektora) do rzeczywistego punktu wejściowego, aby uzyskać wyjaśnienie wpływu cech
    Feature Ablation mierzy wpływ każdej cechy na wynik modelu poprzez sukcesywne usuwanie (ablacja) każdej cechy i obserwowanie zmiany w wyniku modelu.
        
        CNN Models:                     guided gradcam, saliency and feature ablation
        MLP Iris, Wine, Breast cancer:  saliency (barplot)
        MLP Mnist Diff:                 feature_ablation (barplot)
        MLP Mnist Conv:                 saliency (barplot)
    
    """
    loading_state_dict()
    explain_MLP()
    #explain_CNN()
 
  