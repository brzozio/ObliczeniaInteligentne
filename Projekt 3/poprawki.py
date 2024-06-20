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
    'petal width (cm)'
]

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
   

#Atrybucje
def get_attributions(model, input_tensor, target_class, method="saliency", data_offsets=[]):
    model.double()
    model.eval()
    model.to(device)

    if data_offsets != []:
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

def visualize_attributions(attributions, input_tensor, model_name, pred_class=None, prob_class=None, method=None, target_tensor=None, file_name = "debug", table_name=""):

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
        #pred_class = joblib.load(path_script + f"\\debug_temporaries\\{model_name.split()[0]}_{model_name.split()[1]}_pred_targets.joblib")
        #prob_class = joblib.load(path_script + f"\\debug_temporaries\\{model_name.split()[0]}_{model_name.split()[1]}_prob_targets.joblib")
        
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
        
        plt.suptitle(table_name, fontname= 'Arial', fontsize = 30, fontweight = 'bold')
        plt.savefig(path_script+f"/temp/{file_name}.jpg")

def extract_single_class(data_set, target):

    target_class_indices = []

    for obj_id in range(data_set.targets.size(0)):
        if data_set.targets[obj_id].cpu().numpy() == target:
            target_class_indices.append(obj_id)

    return data_set.data[target_class_indices], data_set.targets[target_class_indices], target_class_indices

def post_show_model_biases(target=0):
    
    target_class_data_tensor, target_class_targets, target_class_indices = extract_single_class(data_set = data_CNN_cifar, target = target)

    pred_class = joblib.load(path_script + f"\\debug_temporaries\\CNN_cifar_pred_targets.joblib")
    pred_target_class  = pred_class[target_class_indices]
    prob_class = joblib.load(path_script + f"\\debug_temporaries\\CNN_cifar_prob_targets.joblib")
    prob_target_class  = prob_class[target_class_indices]

    correct_pred_id = []
    incorrect_pred_id = []
    
    for obj_id in range(target_class_targets.size(0)):
        if target_class_targets[obj_id].cpu().numpy() == pred_target_class[obj_id]:
            correct_pred_id.append(obj_id)
        else: 
            incorrect_pred_id.append(obj_id)

    correct_prob_target_class = prob_target_class[correct_pred_id]
    incorrect_prob_target_class = prob_target_class[incorrect_pred_id]

    correct_prob_sort_permutation = np.argsort(correct_prob_target_class)
    incorrect_prob_sort_permutation = np.argsort(incorrect_prob_target_class)

    correct_prob_target_class = correct_prob_target_class[correct_prob_sort_permutation]
    incorrect_prob_target_class = incorrect_prob_target_class[incorrect_prob_sort_permutation]

    correct_pred_target_class = pred_target_class[correct_pred_id]
    correct_pred_target_class = correct_pred_target_class[correct_prob_sort_permutation]
    incorrect_pred_target_class = pred_target_class[incorrect_pred_id]
    incorrect_pred_target_class = incorrect_pred_target_class[incorrect_prob_sort_permutation]

    correct_data_target_class = target_class_data_tensor[correct_pred_id]
    correct_data_target_class = correct_data_target_class[correct_prob_sort_permutation]
    incorrect_data_target_class = target_class_data_tensor[incorrect_pred_id]
    incorrect_data_target_class = incorrect_data_target_class[incorrect_prob_sort_permutation]

    correct_target_class_targets = target_class_targets[correct_pred_id]
    correct_target_class_targets = correct_target_class_targets[correct_prob_sort_permutation]
    incorrect_target_class_targets = target_class_targets[incorrect_pred_id]
    incorrect_target_class_targets = incorrect_target_class_targets[incorrect_prob_sort_permutation]

    correct_size = len(correct_pred_id)
    incorrect_size = len(incorrect_pred_id)

    attributes, input_tensor, target_tensor = get_attributions(model=model_CNN_cifar, input_tensor=correct_data_target_class, 
                        target_class=correct_target_class_targets, method="feature_ablation", data_offsets=[-10,-9,-8,-7,-6,-5,-4,-3,-2,-1])
    visualize_attributions(attributes, input_tensor=input_tensor, model_name="CNN Cifar", method="feature_ablation", 
                        target_tensor=target_tensor, prob_class=correct_prob_target_class[correct_size-10:correct_size], pred_class=correct_pred_target_class[correct_size-10:correct_size], 
                        file_name=f"POST_confident_correct_{cifar10_classes[target]}", table_name=f"Class: {cifar10_classes[target]}, Correctly identified\n with Highest certainty")

    attributes, input_tensor, target_tensor = get_attributions(model=model_CNN_cifar, input_tensor=incorrect_data_target_class, 
                        target_class=incorrect_target_class_targets, method="feature_ablation", data_offsets=[-10,-9,-8,-7,-6,-5,-4,-3,-2,-1])
    visualize_attributions(attributes, input_tensor=input_tensor, model_name="CNN Cifar", method="feature_ablation", 
                        target_tensor=target_tensor, prob_class=incorrect_prob_target_class[incorrect_size-10:incorrect_size], pred_class=incorrect_pred_target_class[incorrect_size-10:incorrect_size], 
                        file_name=f"POST_confident_incorrect_{cifar10_classes[target]}", table_name=f"Class: {cifar10_classes[target]}, Incorrectly identified\n with Highest certainty")

    attributes, input_tensor, target_tensor = get_attributions(model=model_CNN_cifar, input_tensor=correct_data_target_class, 
                        target_class=correct_target_class_targets, method="feature_ablation", data_offsets=[0,1,2,3,4,5,6,7,8,9])
    visualize_attributions(attributes, input_tensor=input_tensor, model_name="CNN Cifar", method="feature_ablation", 
                        target_tensor=target_tensor, prob_class=correct_prob_target_class[0:10], pred_class=correct_pred_target_class[0:10], 
                        file_name=f"POST_shy_correct_{cifar10_classes[target]}", table_name=f"Class: {cifar10_classes[target]}, Correctly identified\n with Lowest certainty")

def get_violin_plot(model, data_set, method, classes, features=None):
    

    attribution, _, _ = get_attributions(model=model, input_tensor=data_set.data, target_class=data_set.targets, method=method)
    attribution = attribution.cpu().numpy()
    max = np.max(attribution) + 1
    min = np.min(attribution) - 1

    _, ax = plt.subplots(len(classes),1)

    for target in range(len(classes)):
        _, _, indicies = extract_single_class(data_set=data_set, target=target)
        ax[target].violinplot(attribution[indicies])
        ax[target].set_title(f"{classes[target]}")
        ax[target].set_ylim([min,max])

    plt.show()

if __name__ == "__main__":

    """
    data_CNN_cifar          = datasets_get.cifar10_to_cnn(device, False)
    model_CNN_cifar         = CNN_tanh(in_side_len=32, in_channels=3, cnv0_out_channels=15, 
                                    cnv1_out_channels=16, lin0_out_size=128, lin1_out_size=10, 
                                    convolution_kernel=7, pooling_kernel=2, reduce_to_dim2=False)
    model_CNN_cifar.load_state_dict(torch.load(path_models + 'CNN_cifar.pth'))
    post_show_model_biases(6)
    """
    
    data_MLP_iris           = datasets_get.iris(device)    
    model_MLP_iris          = MLP(input_size=4, hidden_layer_size=2, classes=3)
    model_MLP_iris.load_state_dict(torch.load(path_models + 'MLP_iris.pth'))

    get_violin_plot(model_MLP_iris, data_MLP_iris, method="feature_ablation", classes=iris_classes)

    """
    data_MLP_wine           = datasets_get.wine(device)
    model_MLP_wine          = MLP(input_size=13, hidden_layer_size=7, classes=3)
    model_MLP_wine.load_state_dict(torch.load(path_models + 'MLP_wine.pth'))
    
    data_MLP_breast_cancer  = datasets_get.breast_cancer(device)
    model_MLP_breast_cancer = MLP(input_size=30, hidden_layer_size=15, classes=2)
    model_MLP_breast_cancer.load_state_dict(torch.load(path_models + 'MLP_breast_cancer.pth'))
    """
    
    
 
  