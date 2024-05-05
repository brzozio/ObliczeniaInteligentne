import torch
from torchvision import datasets, transforms
import numpy as np
import pandas as pd
#import pywt
import cv2

transform = transforms.Compose([
    transforms.ToTensor(),  
])

df = pd.DataFrame({
        'inertia_x': [],
        'inertia_y': [],
        'grad_x': [],
        'grad_y': [],
        'magnitude': [],
        'direction': []
    })
train : bool = True

if __name__ == "__main__":
    if train is True:
        mnist = datasets.MNIST(root='data', train=train, download=True, transform=transform)
        i=0
        for image in mnist.data:
            image_np = image.numpy()
            inertia_x = np.sum((image_np * np.arange(image_np.shape[1])[:, np.newaxis])**2)
            inertia_y = np.sum((image_np * np.arange(image_np.shape[0])[np.newaxis, :])**2)
            grad_x = np.sum(np.abs(cv2.Sobel(image_np, cv2.CV_64F, 1, 0, ksize=3)))
            grad_y = np.sum(np.abs(cv2.Sobel(image_np, cv2.CV_64F, 0, 1, ksize=3)))
            magnitude = np.sum(np.sqrt(grad_x**2 + grad_y**2))
            direction = np.mean(np.arctan2(grad_y, grad_x))

            df = df._append({'inertia_x': inertia_x, 'inertia_y': inertia_y, 'grad_x': grad_x,'grad_y': grad_y,'magnitude': magnitude,'direction': direction}, ignore_index=True)
            print(i)
            i+=1
        df.to_csv(f"extraction_4_train.csv", header=False, index=False)
    else:
        mnist = datasets.MNIST(root='data', train=train, download=True, transform=transform)
        i=0
        for image in mnist.data:
            image_np = image.numpy()
            inertia_x = np.sum((image_np * np.arange(image_np.shape[1])[:, np.newaxis])**2)
            inertia_y = np.sum((image_np * np.arange(image_np.shape[0])[np.newaxis, :])**2)
            grad_x = np.sum(np.abs(cv2.Sobel(image_np, cv2.CV_64F, 1, 0, ksize=3)))
            grad_y = np.sum(np.abs(cv2.Sobel(image_np, cv2.CV_64F, 0, 1, ksize=3)))
            magnitude = np.sum(np.sqrt(grad_x**2 + grad_y**2))
            direction = np.mean(np.arctan2(grad_y, grad_x))
            #print(f"PIC {i} INERTIA X: {inertia_x}")
            #print(f"PIC {i} INERTIA Y: {inertia_y}")
            #print(f"PIC {i} GRAD X: {grad_x}")
            #print(f"PIC {i} GRAD Y: {grad_y}")
            #print(f"PIC {i} MAGNITUDE: {magnitude}")
            #print(f"PIC {i} DIRECTION: {direction}")

            df = df._append({'inertia_x': inertia_x, 'inertia_y': inertia_y, 'grad_x': grad_x,'grad_y': grad_y,'magnitude': magnitude,'direction': direction}, ignore_index=True)
            print(i)
            i+=1
        df.to_csv(f"extraction_4_test.csv", header=False, index=False)

    
