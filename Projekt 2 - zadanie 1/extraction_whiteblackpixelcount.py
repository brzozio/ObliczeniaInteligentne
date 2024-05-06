import torch
from torchvision import datasets, transforms
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import cv2

transform = transforms.Compose([
    transforms.ToTensor(),  
])


train : bool = True

inertia_x_list = []
images         = []

savenumpy_train : np.ndarray = np.ndarray([3000,32])
savenumpy_test  : np.ndarray = np.ndarray([1000,32])


#images_whiteblack : np.ndarray = np.ndarray([16,2]) #7*7 strefy, kazda posiada ilosc white and black pixels
images_white = [] 
images_black = [] 

if __name__ == "__main__":
    if train is True:
        mnist = datasets.MNIST(root='data', train=train, download=True, transform=transform)
        i=0
        for image in mnist.data[1:3000]:
            image_np  = image.numpy()
            #Dzielenie obrazu na 49 części : 4x4 po 7 piksele
            for y in range(0, 28 - 4 + 1, 7):
                for x in range(0, 28 - 4 + 1, 7):
                    fragment = image_np[y:y+4, x:x+4]
                    images.append(fragment)
            images_np = np.array(images)

            for partial_image in images_np:
                threshold = 128  # Próg binaryzacji
                _, binary_image = cv2.threshold(partial_image, threshold, 255, cv2.THRESH_BINARY)
                num_black_pixels = cv2.countNonZero(binary_image)
                num_white_pixels = binary_image.size - num_black_pixels

                images_black.append(num_black_pixels)
                images_white.append(num_white_pixels)
            print(i)
            i+=1

        images_black      = StandardScaler().fit_transform(np.array(images_black).reshape(-1,1))
        images_white      = StandardScaler().fit_transform(np.array(images_white).reshape(-1,1))
       
        
        for row in range(savenumpy_train.shape[0]):
            savenumpy_train[row] = [images_black[0*row][0], images_black[1*row][0], images_black[2*row][0], images_black[3*row][0], images_black[4*row][0], images_black[5*row][0], images_black[6*row][0], images_black[7*row][0], images_black[8*row][0], images_black[9*row][0], images_black[10*row][0], images_black[11*row][0], images_black[12*row][0], images_black[13*row][0], images_black[14*row][0], images_black[15*row][0],images_white[0*row][0], images_white[1*row][0], images_white[2*row][0], images_white[3*row][0], images_white[4*row][0], images_white[5*row][0], images_white[6*row][0], images_white[7*row][0], images_white[8*row][0], images_white[9*row][0], images_white[10*row][0], images_white[11*row][0], images_white[12*row][0], images_white[13*row][0], images_white[14*row][0], images_white[15*row][0]]
        
        np.savetxt('extraction_5_train.txt', savenumpy_train)
    else:
        mnist = datasets.MNIST(root='data', train=train, download=True, transform=transform)
        i=0
        for image in mnist.data[1:1000]:
            image_np  = image.numpy()
            #Dzielenie obrazu na 49 części : 4x4 po 4 piksele
            for y in range(0, 28 - 7 + 1, 4):
                for x in range(0, 28 - 7 + 1, 4):
                    fragment = image_np[y:y+4, x:x+4]
                    images.append(fragment)
            images_np = np.array(images)

            for partial_image in images_np:
                threshold = 128  # Próg binaryzacji
                _, binary_image = cv2.threshold(partial_image, threshold, 255, cv2.THRESH_BINARY)
                num_black_pixels = cv2.countNonZero(binary_image)
                num_white_pixels = binary_image.size - num_black_pixels

                images_black.append(num_black_pixels)
                images_white.append(num_white_pixels)
            print(i)
            i+=1

        images_black      = StandardScaler().fit_transform(np.array(images_black).reshape(-1,1))
        images_white      = StandardScaler().fit_transform(np.array(images_white).reshape(-1,1))
       
        
        for row in range(savenumpy_test.shape[0]):
            savenumpy_test[row] = [images_black[0*row][0], images_black[1*row][0], images_black[2*row][0], images_black[3*row][0], images_black[4*row][0], images_black[5*row][0], images_black[6*row][0], images_black[7*row][0], images_black[8*row][0], images_black[9*row][0], images_black[10*row][0], images_black[11*row][0], images_black[12*row][0], images_black[13*row][0], images_black[14*row][0], images_black[15*row][0],images_white[0*row][0], images_white[1*row][0], images_white[2*row][0], images_white[3*row][0], images_white[4*row][0], images_white[5*row][0], images_white[6*row][0], images_white[7*row][0], images_white[8*row][0], images_white[9*row][0], images_white[10*row][0], images_white[11*row][0], images_white[12*row][0], images_white[13*row][0], images_white[14*row][0], images_white[15*row][0]]
        
        np.savetxt('extraction_5_test.txt', savenumpy_test)

    
