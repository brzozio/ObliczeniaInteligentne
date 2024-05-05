import torch
from torchvision import datasets, transforms
import numpy as np
import pandas as pd

transform = transforms.Compose([
    transforms.ToTensor(),  
])

df = pd.DataFrame({
        'inertia_x': [],
        'inertia_y': []
    })
        
train : bool = False

if __name__ == "__main__":
    if train is True:
        mnist = datasets.MNIST(root='data', train=train, download=True, transform=transform)
        inertia_features = []
        for image in mnist.data:
            image_np = image.numpy()
            
            inertia_x = np.sum((image_np * np.arange(image_np.shape[1])[:, np.newaxis])**2)
            inertia_y = np.sum((image_np * np.arange(image_np.shape[0])[np.newaxis, :])**2)
            
            inertia_features.append([inertia_x, inertia_y])
            df = df._append({'inertia_x': inertia_x, 'inertia_y': inertia_y}, ignore_index=True)

        inertia_features_tensor = torch.tensor(inertia_features)

        print(f'TRAIN INERTIA FEATURES: {inertia_features_tensor}')
        df.to_csv(f"extraction_4_train.csv", header=False, index=False)
    else:
        mnist = datasets.MNIST(root='data', train=train, download=True, transform=transform)
        inertia_features = []
        for image in mnist.data:
            image_np = image.numpy()
            
            inertia_x = np.sum((image_np * np.arange(image_np.shape[1])[:, np.newaxis])**2)
            inertia_y = np.sum((image_np * np.arange(image_np.shape[0])[np.newaxis, :])**2)
            
            inertia_features.append([inertia_x, inertia_y])
            df = df._append({'inertia_x': inertia_x, 'inertia_y': inertia_y}, ignore_index=True)

        inertia_features_tensor = torch.tensor(inertia_features)

        print(f'TEST INERTIA FEATURES: {inertia_features_tensor}')
        df.to_csv(f"extraction_4_test.csv", header=False, index=False)

    
