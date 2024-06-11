import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader

from captum.attr import Saliency
import matplotlib.pyplot as plt
import numpy as np
from model_CNN import CNN_tanh_compose
import os

repo_name = "nteligentne"
path_script = os.path.dirname(os.path.realpath(__file__))
index = path_script.find(repo_name)
path_data = path_script
if index != -1:
   path_data = path_script[:index + len(repo_name)]
   path_data = path_data + "\\data"
path_models = path_script + "\\models\\"



model_CNN_mnist         = CNN_tanh_compose(in_side_len=28, in_channels=1, cnv0_out_channels=12, 
                                   cnv1_out_channels=16, lin0_out_size=100, lin1_out_size=10, 
                                   convolution_kernel=3, pooling_kernel=2, reduce_to_dim2=False)

model_CNN_mnist.load_state_dict(torch.load(path_models + 'CNN_mnist.pth'))
model_CNN_mnist.eval()

# Step 2: Load and preprocess the data
transform = transforms.Compose([
        transforms.ToTensor()
    ])

test_dataset = datasets.MNIST(root=path_data, train=False, transform=transform, download=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Step 3: Select an image from the test dataset
data_iter = iter(test_loader)
images, labels = next(data_iter)
images.requires_grad = True

# Step 4: Calculate saliency using Captum
saliency = Saliency(model_CNN_mnist)
saliency_map = saliency.attribute(images, target=labels)

# Step 5: Visualize the saliency map
def visualize_saliency(img, saliency_map):
    img = img.squeeze().detach().numpy()
    saliency_map = saliency_map.squeeze().detach().numpy()
    
    fig, ax = plt.subplots(1, 2)
    
    ax[0].imshow(img, cmap='gray')
    ax[0].axis('off')
    ax[0].set_title('Original Image')
    
    ax[1].imshow(saliency_map, cmap='hot')
    ax[1].axis('off')
    ax[1].set_title('Saliency Map')
    
    plt.show()

visualize_saliency(images, saliency_map)
