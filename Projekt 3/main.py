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
data_CNN_mnist          = datasets_get.mnist_to_cnn(device, False)
data_CNN_cifar          = datasets_get.cifar10_to_cnn(device, False)

cifar10_classes = [
    "Airplane",
    "Automobile",
    "Bird",
    "Cat",
    "Deer",
    "Dog",
    "Frog",
    "Horse",
    "Ship",
    "Truck"
]
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


def execute_model(data_set, model, data_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'CUDA VERSION: {torch.version.cuda}')
    print(f'DEVICE RUNING: {device}')

    model.load_state_dict(torch.load(path_models + f'{data_name}.pth'))
    model.eval()
    model.double()
    model.to(device)
    outputs = model.forward(data_set.data)

    softmax = torch.nn.Softmax(dim=1)
    probabilities = softmax(outputs)
    predicted_classes = torch.argmax(probabilities, dim=1)

    predicted_classes_cpu   = predicted_classes.cpu().numpy()
    probabilities           = probabilities.cpu().detach().numpy()
    targets_cpu             = data_set.targets.cpu().numpy()
    
    certainty = np.zeros(len(probabilities))
    for i in range(len(certainty)):
        certainty[i] = probabilities[i][predicted_classes_cpu[i]]

    accuracy = accuracy_score(predicted_classes_cpu, targets_cpu)
    print(f'ACCURACY SCORE FOR {data_name}: {accuracy:.4f}')

    joblib.dump(predicted_classes_cpu, path_script + f"\\debug_temporaries\\{data_name}_pred_targets.joblib")
    joblib.dump(certainty, path_script + f"\\debug_temporaries\\{data_name}_prob_targets.joblib")
    

def testing_models_eval():
    #execute_model(data_set=data_CNN_mnist, model=model_CNN_mnist, data_name='CNN_mnist')
    #execute_model(data_set=data_CNN_cifar, model=model_CNN_cifar, data_name='CNN_cifar')
    #execute_model(data_set=data_MLP_iris, model=model_MLP_iris, data_name='MLP_iris')
    #execute_model(data_set=data_MLP_wine, model=model_MLP_wine, data_name='MLP_wine')
    #execute_model(data_set=data_MLP_breast_cancer, model=model_MLP_breast_cancer, data_name='MLP_breast_cancer')
    execute_model(data_set=data_MLP_mnist_conv, model=model_MLP_mnist_conv, data_name='MLP_mnist_extr_conv')
    #execute_model(data_set=data_MLP_mnist_diff, model=model_MLP_mnist_diff, data_name='MLP_mnist_extr_diff')

   
def loading_state_dict():
    model_CNN_mnist.load_state_dict(torch.load(path_models + 'CNN_mnist.pth'))
    model_CNN_cifar.load_state_dict(torch.load(path_models + 'CNN_cifar.pth'))
    model_MLP_iris.load_state_dict(torch.load(path_models + 'MLP_iris.pth'))
    model_MLP_wine.load_state_dict(torch.load(path_models + 'MLP_wine.pth'))
    model_MLP_breast_cancer.load_state_dict(torch.load(path_models + 'MLP_breast_cancer.pth'))
    model_MLP_mnist_conv.load_state_dict(torch.load(path_models + 'MLP_mnist_extr_conv.pth'))
    model_MLP_mnist_diff.load_state_dict(torch.load(path_models + 'MLP_mnist_extr_diff.pth'))

#Atrybucje
def get_attributions(model, input_tensor, target_class, method="saliency", data_offsets=[0]):
    model.double()
    model.eval()
    model.to(device)

    input_tensor = input_tensor[data_offsets].requires_grad_(True)
    target_class = target_class[data_offsets]

    if method == "saliency":
        saliency = Saliency(model)
        attribution = saliency.attribute(input_tensor, target=target_class, abs=False)
    elif method == "guided_gradcam":
        target_layer = model.conv1
        guided_gc = GuidedGradCam(model, target_layer)
        attribution = guided_gc.attribute(input_tensor, target=target_class)
    elif method == "lime":
        lime = Lime(model)
        attribution = lime.attribute(input_tensor, target=target_class)
    elif method == "feature_ablation":
        ftr_abl = FeatureAblation(model)
        attribution = ftr_abl.attribute(input_tensor, target=target_class)
    elif method == "integrated_gradients":
        integrated_gradients = IntegratedGradients(model)
        attribution = integrated_gradients.attribute(input_tensor, target=target_class)
    elif method == "shapley":
        shapley_value_sampling = ShapleyValueSampling(model)
        attribution = shapley_value_sampling.attribute(input_tensor, target=target_class, n_samples=2)
    elif method == "deeplift":
        deepliftshapval = DeepLiftShap(model)
        attribution = deepliftshapval.attribute(input_tensor, target=target_class, baselines=torch.zeros(input_tensor.size(), device=device))
    else:
        raise ValueError(f"Unknown method was specified: {method}")

    #print(f'ATTRIBUTION for {method} is: {attribution}, shape: {attribution.shape}, size: {attribution.dim}')
    return attribution, input_tensor, target_class

def tensor_to_attribution_heatmap(tensor):
    out = tensor.cpu().detach()
    for channel in range(1, out.size(0), 1):
        out[0] += out[channel]
    out = out[0]
    return out



def visualize_attributions(attributions, input_tensor, model_name, method=None, target_tensor=None):

    matplotlib.rcParams.update({'font.size': 7})

    if method == "saliency_barplot" or method == "integrated_gradients_barplot" or method ==  "feature_ablation_barplot": #Bar-plot
        #WORKING
        if model_name.split()[1] == "Mnist":
            if input_tensor.size(0) != 10:
                print(f"error, expected batch size = 10, got batch size = {input_tensor.size(0)}")
                return
             
            pred_class = joblib.load(path_script + f"\\debug_temporaries\\{model_name.split()[0]}_{model_name.split()[1]}_{model_name.split()[2]}_pred_targets.joblib")
            prob_class = joblib.load(path_script + f"\\debug_temporaries\\{model_name.split()[0]}_{model_name.split()[1]}_{model_name.split()[2]}_prob_targets.joblib")
            _, ax = plt.subplots(5, 4, figsize=(10, 5))
            plt.subplots_adjust(hspace=0.55)

            print(f'Pred class: {pred_class[0]}, orig: {target_tensor[0]} for model: {model_name.split()[0]}_{model_name.split()[1]}_{model_name.split()[2]}_pred_targets')
        
            for index in range(target_tensor.size(0)):
                sb.barplot(x=[0,1,2,3,4,5,6,7,8,9], y=attributions[index].cpu().detach().numpy(), ax=ax[index//2,(index%2)*2+1])
                ax[index//2,(index%2)*2+1].set_title(f'Predicted: [{pred_class[index]}] Orig: [{target_tensor[index]}], Probability: [{prob_class[index]:.4f}]', fontsize=12, fontweight = 'bold')
                ax[index//2,(index%2)*2].imshow(input_tensor[index][0].cpu().detach().numpy())
                ax[index//2,(index%2)*2].tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=False)
                ax[index//2,(index%2)*2].tick_params(axis='y',which='both',left=False,right=False,labelleft=False)
                ax[index//2,(index%2)*2+1].set_ylabel('Attribution')
                ax[index//2,(index%2)*2+1].set_ylim([np.min(attributions.cpu().detach().numpy().flatten()), np.max(attributions.cpu().detach().numpy().flatten())])
    
            plt.xlabel('Feature')
            if method == "saliency_barplot":
                plt.suptitle(f"xAI for {model_name}, Method: {method.split(sep='_')[0]}", fontname= 'Arial', fontsize = 30, fontweight = 'bold')
            else:
                plt.suptitle(f"xAI for {model_name}, Method: {method.split(sep='_')[0]} {method.split(sep='_')[1]}", fontname= 'Arial', fontsize = 30, fontweight = 'bold')

            plt.show()
            return


        _, ax = plt.subplots(attributions.size(0), 1, figsize=(10, 5))

        class_alias = range(target_tensor.size(0))
        feature_alias = range(torch.numel(attributions)//attributions.size(0))

        if model_name.split()[1] == "iris":
            class_alias = iris_classes
            feature_alias = iris_features            
        elif model_name.split()[1] == "wine":
            class_alias = wine_classes
            feature_alias = wine_features
        elif model_name.split()[1] == "breast":
            class_alias = breast_cancer_classes
            feature_alias = breast_cancer_features
            for i in range(attributions.size(0)):
                ax[i].tick_params(axis='x', rotation=90)
        
        pred_class = joblib.load(path_script + f"\\debug_temporaries\\{model_name.split()[0]}_{model_name.split()[1]}_pred_targets.joblib")
        prob_class = joblib.load(path_script + f"\\debug_temporaries\\{model_name.split()[0]}_{model_name.split()[1]}_prob_targets.joblib")

        for i in range(attributions.size(0)):
            sb.barplot(x=feature_alias, y=attributions[i].cpu().detach().numpy(), ax=ax[i])
            ax[i].set_ylabel('Attribution')
            ax[i].set_ylim([np.min(attributions.cpu().detach().numpy().flatten()), np.max(attributions.cpu().detach().numpy().flatten())])       
            ax[i].set_title(f'Predicted: [{class_alias[pred_class[i]]}], Orig: [{class_alias[target_tensor[i].item()]}], Probability: [{prob_class[i]:.4f}]', fontsize=12, fontweight = 'bold')
            if i != attributions.size(0) - 1:                
                ax[i].tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=False)
            
        plt.xlabel('Feature')
        if method == "saliency_barplot":
            plt.suptitle(f"xAI for {model_name}, Method: {method.split(sep='_')[0]}", fontname= 'Arial', fontsize = 30, fontweight = 'bold')
        else:
            plt.suptitle(f"xAI for {model_name}, Method: {method.split(sep='_')[0]} {method.split(sep='_')[1]}", fontname= 'Arial', fontsize = 30, fontweight = 'bold')

        plt.show()

    elif method == "diff_feature_ablation" or method == "diff_saliency_map":
        pred_class = joblib.load(path_script + f"\\debug_temporaries\\{model_name.split()[0]}_{model_name.split()[1]}_{model_name.split()[2]}_pred_targets.joblib")
        prob_class = joblib.load(path_script + f"\\debug_temporaries\\{model_name.split()[0]}_{model_name.split()[1]}_{model_name.split()[2]}_prob_targets.joblib")
        
        fig_min = np.min(attributions.cpu().detach().numpy().flatten()) - 0.1
        fig_max = np.max(attributions.cpu().detach().numpy().flatten()) + 0.1
        

        for example in range(input_tensor.size(0)):
            print(f"pred class: {pred_class[example]} for model {model_name.split()[0]}_{model_name.split()[1]}_{model_name.split()[2]}_pred_targets.joblib")
            fig = plt.figure(figsize=(5, 5))
            gs = GridSpec(4, 4, figure=fig) 
            # image
            ax1 = fig.add_subplot(gs[0:3, 0:3])
            ax1.imshow(input_tensor[example][0].cpu().detach().numpy())
            ax1.set_title(f"predicted class: {pred_class[example]}, orig class: {target_tensor[example]}", fontname= 'Arial', fontsize = 10, fontweight = 'bold')
            # vertical axis - 28:56
            ax2 = fig.add_subplot(gs[0:3, 3])
            ax2.plot(attributions[example][28:56].cpu().detach().numpy(), range(28))
            ax2.set_title(f"probability: {prob_class[example] :.4f}", fontsize = 10, fontweight = 'bold')
            # horizontal axis = 0:28
            ax3 = fig.add_subplot(gs[3, 0:3])
            ax3.plot(attributions[example][0:28].cpu().detach().numpy())
            
            ax2.set_xlim([fig_min,fig_max])
            ax3.set_ylim([fig_min,fig_max])
            ax3.invert_yaxis()
            ax1.tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=False)
            ax1.tick_params(axis='y',which='both',left=False,right=False,labelleft=False)
            ax2.tick_params(axis='y',which='both',left=False,right=True,labelleft=False)
            ax3.tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=False)
            ax3.yaxis.tick_right()

            plt.suptitle(f"xAI for {model_name}, Method: {method.split(sep='_')[1]} {method.split(sep='_')[2]}", fontname='Arial', fontsize=15, fontweight='bold')
            plt.subplots_adjust(wspace=0.1, hspace=0.1)
            plt.savefig(path_script+f"/temp/{model_name}_{method}_{target_tensor[example]}.jpg")
        
    else:
        if input_tensor.size(0) != 10:
            print(f"error, expected batch size = 10, got batch size = {input_tensor.size(0)}")
            return
        #WORKING
        _, ax = plt.subplots(5,4, figsize=(10,14))
        pred_class = joblib.load(path_script + f"\\debug_temporaries\\{model_name.split()[0]}_{model_name.split()[1]}_pred_targets.joblib")
        prob_class = joblib.load(path_script + f"\\debug_temporaries\\{model_name.split()[0]}_{model_name.split()[1]}_prob_targets.joblib")
        
        class_mapping = range(10)
        if model_name.split()[1] == "Cifar":
            class_mapping = cifar10_classes
        
        for i in range(10):
            ax[i//2,(i%2)*2].set_title(f"predicted class: {class_mapping[pred_class[i]]}", fontsize = 10, fontweight = 'bold')
            ax[i//2,(i%2)*2+1].set_title(f"probability: {prob_class[i] :.4f}", fontsize = 10, fontweight = 'bold')


        format_to_im = lambda tensor : \
            tensor.cpu().detach().numpy().transpose(1,2,0)/255
        
        fig_min = abs(np.min(attributions.cpu().detach().numpy().flatten()))
        fig_max = abs(np.max(attributions.cpu().detach().numpy().flatten()))
        if fig_max < fig_min: fig_max = fig_min

        for i in range(10):
            ax[i//2,(i%2)*2].imshow(format_to_im(input_tensor[i]))
            ax[i//2,(i%2)*2+1].imshow(tensor_to_attribution_heatmap(attributions[i])/fig_max, cmap='seismic', vmin=-1.0, vmax=1.0)
            
        for i in range(5):
            for j in range(4):
                ax[i,j].tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=False)
                ax[i,j].tick_params(axis='y',which='both',left=False,right=False,labelleft=False)
        
        plt.suptitle(f"xAI for {model_name}, Method: {method}", fontname= 'Arial', fontsize = 30, fontweight = 'bold')
        plt.savefig(path_script+f"/temp/{model_name}_{method}.jpg")




def explain_CNN():

    mnist_examples = [3,5,1,32,4,8,98,36,84,7]
    cifar_examples = [0,5,8,13,67,15,17,32,45,23]

    # ablation, input_tensor, target_tensor  = get_attributions(model=model_CNN_cifar, input_tensor=data_CNN_cifar.data, target_class=data_CNN_cifar.targets, method="feature_ablation", data_offsets=cifar_examples)
    # visualize_attributions(ablation, input_tensor, model_name="CNN Cifar",  method="feature_ablation", target_tensor=target_tensor)
    
    # ablation, input_tensor, target_tensor  = get_attributions(model=model_CNN_mnist, input_tensor=data_CNN_mnist.data, target_class=data_CNN_mnist.targets, method="feature_ablation", data_offsets=mnist_examples)
    # visualize_attributions(ablation, input_tensor, model_name="CNN Mnist",  method="feature_ablation", target_tensor=target_tensor)
   
    # saliency, input_tensor, target_tensor  = get_attributions(model=model_CNN_cifar, input_tensor=data_CNN_cifar.data, target_class=data_CNN_cifar.targets, method="saliency", data_offsets=cifar_examples)   
    # visualize_attributions(saliency, input_tensor, model_name="CNN Cifar",  method="saliency_image", target_tensor=target_tensor)
    
    # saliency, input_tensor, target_tensor  = get_attributions(model=model_CNN_mnist, input_tensor=data_CNN_mnist.data, target_class=data_CNN_mnist.targets, method="saliency", data_offsets=mnist_examples)
    # visualize_attributions(saliency, input_tensor, model_name="CNN Mnist",  method="saliency_image", target_tensor=target_tensor)

    # gradcam_attr, input_tensor, target_tensor  = get_attributions(model=model_CNN_cifar, input_tensor=data_CNN_cifar.data, target_class=data_CNN_cifar.targets, method="guided_gradcam", data_offsets=cifar_examples)
    # visualize_attributions(gradcam_attr, input_tensor, model_name="CNN Cifar",  method="guided_gradcam", target_tensor=target_tensor)
    
    # gradcam_attr, input_tensor, target_tensor  = get_attributions(model=model_CNN_mnist, input_tensor=data_CNN_mnist.data, target_class=data_CNN_mnist.targets, method="guided_gradcam", data_offsets=mnist_examples)
    # visualize_attributions(gradcam_attr, input_tensor, model_name="CNN Mnist",  method="guided_gradcam", target_tensor=target_tensor)
    
    # shap_attr, input_tensor, target_tensor  = get_attributions(model=model_CNN_cifar, input_tensor=data_CNN_cifar.data, target_class=data_CNN_cifar.targets, method="shapley", data_offsets=cifar_examples)
    # visualize_attributions(shap_attr, input_tensor, model_name="CNN Cifar",  method="guided_gradcam", target_tensor=target_tensor)
    
    # shap_attr, input_tensor, target_tensor  = get_attributions(model=model_CNN_mnist, input_tensor=data_CNN_mnist.data, target_class=data_CNN_mnist.targets, method="guided_gradcam", data_offsets=mnist_examples)
    # visualize_attributions(shap_attr, input_tensor, model_name="CNN Mnist",  method="guided_gradcam", target_tensor=target_tensor)
    
    shap_attr, input_tensor, target_tensor = get_attributions(model=model_CNN_cifar, input_tensor=data_CNN_cifar.data, target_class=data_CNN_cifar.targets, method="shapley", data_offsets=cifar_examples)
    visualize_attributions(shap_attr, input_tensor=input_tensor, model_name="CNN Cifar",  method="shapley", target_tensor=target_tensor)
    
    shap_attr, input_tensor, target_tensor = get_attributions(model=model_CNN_mnist, input_tensor=data_CNN_mnist.data, target_class=data_CNN_mnist.targets, method="shapley", data_offsets=mnist_examples)
    visualize_attributions(shap_attr, input_tensor=input_tensor, model_name="CNN Mnist",  method="shapley", target_tensor=target_tensor)


def explain_MLP():
    
    mnist_examples = [3,5,1,32,4,8,98,36,84,7]

    saliency_attr, input_tensor, target_tensor = get_attributions(model=model_MLP_iris, input_tensor=data_MLP_iris.data, target_class=data_MLP_iris.targets, method="saliency", data_offsets=[0,50,100])
    visualize_attributions(saliency_attr, input_tensor=input_tensor, model_name="MLP iris",  method="saliency_barplot", target_tensor=target_tensor)
    
    # saliency_attr, input_tensor, target_tensor = get_attributions(model=model_MLP_wine, input_tensor=data_MLP_wine.data, target_class=data_MLP_wine.targets, method="saliency", data_offsets=[0,60,130])
    # visualize_attributions(saliency_attr, input_tensor=input_tensor, model_name="MLP wine",  method="saliency_barplot", target_tensor=target_tensor)
    
    # saliency_attr, input_tensor, target_tensor = get_attributions(model=model_MLP_breast_cancer, input_tensor=data_MLP_breast_cancer.data, target_class=data_MLP_breast_cancer.targets, method="saliency", data_offsets=[0,21])
    # visualize_attributions(saliency_attr, input_tensor=input_tensor, model_name="MLP breast cancer", target_tensor=target_tensor, method="saliency_barplot")

    # int_grd, input_tensor, target_tensor  = get_attributions(model=model_MLP_iris, input_tensor=data_MLP_iris.data, target_class=data_MLP_iris.targets, method="integrated_gradients", data_offsets=, data_offsets=[0,50,100])
    # visualize_attributions(int_grd, input_tensor, model_name="MLP iris",  method="integrated_gradients_barplot", target_tensor=target_tensor)
    
    # int_grd, input_tensor, target_tensor  = get_attributions(model=model_MLP_wine, input_tensor=data_MLP_wine.data, target_class=data_MLP_wine.targets, method="integrated_gradients", data_offsets=, data_offsets=[0,60,130])
    # visualize_attributions(int_grd, input_tensor, model_name="MLP wine",  method="integrated_gradients_barplot", target_tensor=target_tensor)

    # int_grd, input_tensor, target_tensor  = get_attributions(model=model_MLP_breast_cancer, input_tensor=data_MLP_breast_cancer.data, target_class=data_MLP_breast_cancer.targets, method="integrated_gradients", data_offsets=[0,21])
    # visualize_attributions(int_grd, input_tensor, model_name="MLP breast cancer", target_tensor=target_tensor, method="integrated_gradients_barplot")
    
    saliency_attr, _, _ = get_attributions(model=model_MLP_mnist_diff, input_tensor=data_MLP_mnist_diff.data, target_class=data_MLP_mnist_diff.targets, method="saliency", data_offsets=mnist_examples)
    input_tensor = data_CNN_mnist.data[mnist_examples]
    target_tensor = data_CNN_mnist.targets[mnist_examples]
    visualize_attributions(saliency_attr, input_tensor=input_tensor, model_name="MLP Mnist Diff",  method="diff_saliency_map", target_tensor=target_tensor)

    #saliency_attr, _, _  = get_attributions(model=model_MLP_mnist_diff, input_tensor=data_MLP_mnist_diff.data, target_class=data_MLP_mnist_diff.targets, method="feature_ablation", data_offsets=mnist_examples)
    #input_tensor = data_CNN_mnist.data[mnist_examples]
    #target_tensor = data_CNN_mnist.targets[mnist_examples]
    #visualize_attributions(saliency_attr, input_tensor, model_name="MLP Mnist Diff",  method="diff_feature_ablation", target_tensor=target_tensor)
    
    # saliency_attr, _, _  = get_attributions(model=model_MLP_mnist_conv, input_tensor=data_MLP_mnist_conv.data, target_class=data_MLP_mnist_conv.targets, method="saliency", data_offsets=mnist_examples)
    # input_tensor = data_CNN_mnist.data[mnist_examples]
    # target_tensor = data_CNN_mnist.targets[mnist_examples]
    #visualize_attributions(saliency_attr, input_tensor, model_name="MLP Mnist Conv",  method="saliency_barplot", target_tensor=target_tensor)
    
    # saliency_attr, _, _  = get_attributions(model=model_MLP_mnist_conv, input_tensor=data_MLP_mnist_conv.data, target_class=data_MLP_mnist_conv.targets, method="feature_ablation", data_offsets=mnist_examples)
    # input_tensor = data_CNN_mnist.data[mnist_examples]
    # target_tensor = data_CNN_mnist.targets[mnist_examples]
    # visualize_attributions(saliency_attr, input_tensor, model_name="MLP Mnist Conv",  method="feature_ablation_barplot", target_tensor=target_tensor)



if __name__ == "__main__":
    #testing_models_eval()
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
   # explain_CNN()
 
  