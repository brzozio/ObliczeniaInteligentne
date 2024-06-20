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

model_CNN_cifar         = CNN_tanh(in_side_len=32, in_channels=3, cnv0_out_channels=15, 
                                   cnv1_out_channels=16, lin0_out_size=128, lin1_out_size=10, 
                                   convolution_kernel=7, pooling_kernel=2, reduce_to_dim2=False)

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
   
def loading_state_dict():
    model_CNN_cifar.load_state_dict(torch.load(path_models + 'CNN_cifar.pth'))

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

def visualize_attributions(attributions, input_tensor, model_name, pred_class=None, prob_class=None, method=None, target_tensor=None):

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
        
        plt.suptitle(f"xAI for {model_name}, Method: {method}", fontname= 'Arial', fontsize = 30, fontweight = 'bold')
        plt.savefig(path_script+f"/temp/{model_name}_{method}.jpg")

def explain_CNN():

    mnist_examples = [3,5,1,32,4,8,98,36,84,7]
   
    bird_data = []
    bird_indices = []

    for idx, (img, label) in enumerate(zip(data_CNN_cifar.data, data_CNN_cifar.targets)):
        if label == 2:
            bird_data.append(torch.tensor(img, device=device))
            bird_indices.append(idx)
    bird_data_tensor = torch.stack(bird_data)
    birds_targets = torch.tensor([label for img, label in zip(data_CNN_cifar.data, data_CNN_cifar.targets) if label == 2], device=device)

    pred_class = joblib.load(path_script + f"\\debug_temporaries\\CNN_cifar_pred_targets.joblib")
    pred_bird  = pred_class[bird_indices]
    prob_class = joblib.load(path_script + f"\\debug_temporaries\\CNN_cifar_prob_targets.joblib")
    prob_bird  = prob_class[bird_indices]

    shap_attr, input_tensor, target_tensor = get_attributions(model=model_CNN_cifar, input_tensor=bird_data_tensor, target_class=birds_targets, method="feature_ablation", data_offsets=mnist_examples)
    visualize_attributions(shap_attr, input_tensor=input_tensor, model_name="CNN Cifar",  method="feature_ablation", target_tensor=target_tensor, prob_class=prob_bird, pred_class=pred_bird)

if __name__ == "__main__":

    loading_state_dict()
    explain_CNN()
 
  