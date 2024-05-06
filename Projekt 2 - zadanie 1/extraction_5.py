import torch
from torchvision import datasets, transforms
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
#import pywt
import cv2

transform = transforms.Compose([
    transforms.ToTensor(),  
])

#df = pd.DataFrame({
#        'inertia_x': [],
#        'inertia_y': [],
#        'grad_x': [],
#        'grad_y': [],
#        'magnitude': [],
#        'direction': []
#    })
train : bool = False

inertia_x_list          = []
inertia_y_list          = []
grad_x_list             = []
grad_y_list             = []
magnitude_list          = []
direction_list          = []
num_black_pixels_list   = []
num_white_pixels_list   = []

savenumpy_train : np.ndarray = np.ndarray([60000,8])
savenumpy_test  : np.ndarray = np.ndarray([10000,8])


if __name__ == "__main__":
    if train is True:
        mnist = datasets.MNIST(root='data', train=train, download=True, transform=transform)
        i=0
        for image in mnist.data:
            image_np  = image.numpy()
            inertia_x = np.sum((image_np * np.arange(image_np.shape[1])[:, np.newaxis])**2)/28*28
            inertia_y = np.sum((image_np * np.arange(image_np.shape[0])[np.newaxis, :])**2)/28*28
            grad_x    = np.sum(np.abs(cv2.Sobel(image_np, cv2.CV_64F, 1, 0, ksize=5)))/28*28
            grad_y    = np.sum(np.abs(cv2.Sobel(image_np, cv2.CV_64F, 0, 1, ksize=5)))/28*28
            magnitude = np.sum(np.sqrt(grad_x**2 + grad_y**2))
            direction = np.mean(np.arctan2(grad_y, grad_x))

            #ZLiczanie ilosci bialych i czarnych pikseli
            threshold = 128  # Próg binaryzacji
            _, binary_image = cv2.threshold(image_np, threshold, 255, cv2.THRESH_BINARY)
            num_black_pixels = cv2.countNonZero(binary_image)
            num_white_pixels = binary_image.size - num_black_pixels

            inertia_x_list.append(inertia_x)
            inertia_y_list.append(inertia_y)
            grad_x_list.append(grad_x)
            grad_y_list.append(grad_y)
            magnitude_list.append(magnitude)
            direction_list.append(direction)    
            num_black_pixels_list.append(num_black_pixels) 
            num_white_pixels_list.append(num_white_pixels) 
            print(i)
            i+=1

        inertia_x_list             = StandardScaler().fit_transform(np.array(inertia_x_list).reshape(-1,1))
        inertia_y_list             = StandardScaler().fit_transform(np.array(inertia_y_list).reshape(-1,1))
        grad_x_list                = StandardScaler().fit_transform(np.array(grad_x_list).reshape(-1,1))
        grad_y_list                = StandardScaler().fit_transform(np.array(grad_y_list).reshape(-1,1))
        magnitude_list             = StandardScaler().fit_transform(np.array(magnitude_list).reshape(-1,1))
        direction_list             = StandardScaler().fit_transform(np.array(direction_list).reshape(-1,1))
        num_black_pixels_list      = StandardScaler().fit_transform(np.array(num_black_pixels_list).reshape(-1,1))
        num_white_pixels_list      = StandardScaler().fit_transform(np.array(num_white_pixels_list).reshape(-1,1))
        
        #for index in range(len(inertia_x_list)):
            #df = df._append({'inertia_x': inertia_x_list[index][0], 'inertia_y': inertia_y_list[index][0], 'grad_x': grad_x_list[index][0],'grad_y': grad_y_list[index][0],'magnitude': magnitude_list[index][0],'direction': direction_list[index][0]}, ignore_index=True)
        
        for row in range(savenumpy_train.shape[0]):
            savenumpy_train[row] = [inertia_x_list[row][0], inertia_y_list[row][0], grad_x_list[row][0], grad_y_list[row][0], magnitude_list[row][0], direction_list[row][0], num_white_pixels_list[row][0], num_black_pixels_list[row][0]]
        
        np.savetxt('extraction_5_train.txt', savenumpy_train)
        #df.to_csv(f"extraction_5_train.csv", header=False, index=False)
    else:
        mnist = datasets.MNIST(root='data', train=train, download=True, transform=transform)
        i=0
        for image in mnist.data:
            image_np  = image.numpy()
            inertia_x = np.mean((image_np * np.arange(image_np.shape[1])[:, np.newaxis])**2)
            inertia_y = np.mean((image_np * np.arange(image_np.shape[0])[np.newaxis, :])**2)
            grad_x    = np.mean(np.abs(cv2.Sobel(image_np, cv2.CV_64F, 1, 0, ksize=5)))
            grad_y    = np.mean(np.abs(cv2.Sobel(image_np, cv2.CV_64F, 0, 1, ksize=5)))
            magnitude = np.mean(np.sqrt(grad_x**2 + grad_y**2))
            direction = np.mean(np.arctan2(grad_y, grad_x))

            #ZLiczanie ilosci bialych i czarnych pikseli
            #Dzielenie obrazu na 49 części : 7x7 po 4 piksele
            threshold = 128  # Próg binaryzacji
            _, binary_image = cv2.threshold(image_np, threshold, 255, cv2.THRESH_BINARY)
            num_black_pixels = cv2.countNonZero(binary_image)
            num_white_pixels = binary_image.size - num_black_pixels
            print(f"WHITE: {num_white_pixels}")
            print(f"BLACK: {num_black_pixels}")

            inertia_x_list.append(inertia_x)
            inertia_y_list.append(inertia_y)
            grad_x_list.append(grad_x)
            grad_y_list.append(grad_y)
            magnitude_list.append(magnitude)
            direction_list.append(direction)    
            num_black_pixels_list.append(num_black_pixels) 
            num_white_pixels_list.append(num_white_pixels) 

            print(i)
            i+=1

        inertia_x_list      = StandardScaler().fit_transform(np.array(inertia_x_list).reshape(-1,1))
        inertia_y_list      = StandardScaler().fit_transform(np.array(inertia_y_list).reshape(-1,1))
        grad_x_list         = StandardScaler().fit_transform(np.array(grad_x_list).reshape(-1,1))
        grad_y_list         = StandardScaler().fit_transform(np.array(grad_y_list).reshape(-1,1))
        magnitude_list      = StandardScaler().fit_transform(np.array(magnitude_list).reshape(-1,1))
        direction_list      = StandardScaler().fit_transform(np.array(direction_list).reshape(-1,1))
        num_black_pixels_list      = StandardScaler().fit_transform(np.array(num_black_pixels_list).reshape(-1,1))
        num_white_pixels_list      = StandardScaler().fit_transform(np.array(num_white_pixels_list).reshape(-1,1))
        
        for row in range(savenumpy_test.shape[0]):
            savenumpy_train[row] = [inertia_x_list[row][0], inertia_y_list[row][0], grad_x_list[row][0], grad_y_list[row][0], magnitude_list[row][0], direction_list[row][0], num_white_pixels_list[row][0], num_black_pixels_list[row][0]]
        
        np.savetxt('extraction_5_test.txt', savenumpy_test)

    
