import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier as MLP
from voronoi import plot_decision_boundary, plot_decision_boundary_ax
from sklearn.metrics import accuracy_score
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import torch
from exe_model import CNN
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch import load as load_model
from torch import save as save_model


class CustomDataset(Dataset):
    def __init__(self, data, targets, device):
        self.data = torch.tensor(data, dtype=torch.double, device=device)
        self.targets = torch.tensor(targets, dtype=torch.long, device=device)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = {'data': self.data[idx], 'target': self.targets[idx]}
        return sample

basic = transforms.Compose([
    transforms.ToTensor()
])

flip = transforms.Compose([
    #Odwrócenie obrazu horyzontalnie - w przypadku liczb bez sensu
    transforms.RandomHorizontalFlip(),    
    transforms.ToTensor(),                 
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  
])

rotate = transforms.Compose([
    #Rotacja obrazu o rand kąt <0,15> 
    transforms.RandomRotation(15),          
    transforms.ToTensor(),                  
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  
])

color_jitter = transforms.Compose([
    #Modulacja jasności, kontrastu, nasycenia w obrazie - sens, zdjęcia mogą być ciemniejsze itd.
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  
    transforms.ToTensor(),                                                           
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))       
])

random_crop = transforms.Compose([
    #Randomowe przycięcie obrazu - moze np. wyciąć samą głowę zwierzęcia
    transforms.RandomCrop(32, padding=4),                   
    transforms.ToTensor(),                                  
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def augmenting_image_ax():
    # Zaprezentowac zdjęcie przed i po augmentacji
    mnist = datasets.MNIST(
            root='data',
            train=True,
            download=True,
            transform= transforms.ToTensor()
    )
    _, ax = plt.subplots(3,2, figsize=(10,20))
    
    for row in range(3):
        for col in range(2):
            image = mnist[(row+col)*2][0][0]
            ax[row,col].imshow(image)
            ax[row,col].set_title(f'Liczba przed augmetnacją')
            
            augmented_image = flip(image)
            ax[row,col].imshow(augmented_image)
            ax[row,col].set_title(f'Liczba po augmetnacji')

    # a w przypadku 2 cech należy zaprezentować rozkład danych treningowych w przypadku gdy
    # rozważane było 100 przykładów raz gdy tylko te dane są widoczne i
    # raz gdy dla każdej danej 10 razy została zastosowana metoda augmentacji (1000 przykładów).



def mnist_to_cnn(device, train, transforming) -> CustomDataset:
    mnist           = datasets.MNIST(root='data', train=train, download=True, transform=transforming)
    mnists          = CustomDataset(data=mnist.data, targets=mnist.targets, device=device)
    mnists.data     = mnists.data.view(-1,1,28,28)
    return mnists

def run_random_state(reduce_dim) -> None:
    print(f'RUNNING: {device}')
    df_avg_acc = pd.DataFrame({
        'all': [],
        '100': [],
        '200': [],
        '1000': []
    })
    df_std_div_acc = pd.DataFrame({
        'all': [],
        '100': [],
        '200': [],
        '1000': []
    })
    
    model = CNN(in_side_len=28, in_channels=1, cnv0_out_channels=8, cnv1_out_channels=16,
                        reduce_to_dim2=reduce_dim, lin0_out_size=20, lin1_out_size=10,
                        convolution_kernel=5, pooling_kernel=2)
    
    augmentations = [basic, rotate, color_jitter]
    sample_sizes = [100,200,1000,60000]
    criteria     = torch.nn.CrossEntropyLoss()
    optimizer    = torch.optim.Adam(model.parameters(), lr=0.01)
    num_epochs   = 50
    
    avg_acc_aug           = np.array([])
    std_acc_aug           = np.array([])

    for augm_i, augm in enumerate(augmentations):
        data_set_train        = mnist_to_cnn(device, True, augm)
        data_set_basic_test   = mnist_to_cnn(device, False, basic)


        for sample_size in sample_sizes: 
            print(f"SAMPLE SIZE: {sample_size}")
            accuracy_score_list = np.array([])
            max_accuracy        = 0
               
            sample_param = torch.randperm(len(data_set_train))[:sample_size]
            dataloader = DataLoader(dataset=data_set_train, batch_size=10, sampler=SubsetRandomSampler(sample_param))

            for run in range(10):
                model = CNN(in_side_len=28, in_channels=1, cnv0_out_channels=8, cnv1_out_channels=16,
                        reduce_to_dim2=reduce_dim, lin0_out_size=20, lin1_out_size=10,
                        convolution_kernel=5, pooling_kernel=2)
                #Trenowanie modelu w kazdym run - 100, 200, 1000, All dane 
                model.train()
                model.double()
                model.to(device)
                #print(f'DATA SIZE: {data_set_train.data.size()}')

                for epoch in range(num_epochs):
                    for batch in dataloader:
                        data, target = batch['data'].to(device), batch['target'].to(device)
                        outputs = model.extract(data)
                        outputs = model.forward(outputs)
                        loss = criteria(outputs, target)
                        
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    
                    print(f"Epoch [{epoch+1}/{num_epochs}]  Loss: {loss.item():.5f} - RUN [{run}]")

                model.eval()
                model.double()
                model.to(device)

                outputs = model.extract(data_set_basic_test.data)
                outputs = model.forward(outputs)
                #print(f"OUTPUS: {outputs}")
                
                softmax = torch.nn.Softmax(dim=1)
                probabilities = softmax(outputs)
                predicted_classes = torch.argmax(probabilities, dim=1)

                predicted_classes_cpu = predicted_classes.cpu().numpy()
                targets_cpu           = data_set_basic_test.targets.cpu().numpy()

                #print(f'PREDICTED CLASSES: {predicted_classes}')
                #print(f"ORIGINAL CLASSES: {data_set_basic_test.targets}")
                
                accuracy = accuracy_score(predicted_classes_cpu, targets_cpu)
                accuracy_score_list = np.append(accuracy_score_list,accuracy)
                #print(f'ACCURACY SCORE: {accuracy:.4f}')
                
                if accuracy > max_accuracy:
                    max_accuracy = accuracy
                    save_model(model.state_dict(), f'RUN_10_TIMES_{augm_i}_{sample_size}_MNIST.pth') 
                
            #Wyliczanie sredniej acc i odchylenie standardowe acc dla test
            avg_acc = accuracy_score_list.mean()
            std_div = accuracy_score_list.std()

            avg_acc_aug = np.append(avg_acc_aug, avg_acc) 
            print(f'SIZE AVG: {avg_acc_aug.size}')
            std_acc_aug = np.append(std_acc_aug, std_div)
            print(f'SIZE AVG: {std_acc_aug.size}')

        
    #3 wiersze dla kazdej tabeli - brak augmentacji, augmentacja 1, augmentacja 2
    new_row_run_avg_acc_no_aug = { 
                            'all': avg_acc_aug[3], 
                            '100': avg_acc_aug[0], 
                            '200': avg_acc_aug[1], 
                            '1000': avg_acc_aug[2]
    }

    new_row_run_avg_acc_aug_1 = {
                            'all': avg_acc_aug[7], 
                            '100': avg_acc_aug[4], 
                            '200': avg_acc_aug[5], 
                            '1000': avg_acc_aug[6]
    }
    new_row_run_avg_acc_aug_2 = { 
                            'all': avg_acc_aug[11], 
                            '100': avg_acc_aug[8], 
                            '200': avg_acc_aug[9], 
                            '1000': avg_acc_aug[10]
    }
    
    
    df_avg_acc = df_avg_acc._append(new_row_run_avg_acc_no_aug, ignore_index=True)
    df_avg_acc = df_avg_acc._append(new_row_run_avg_acc_aug_1, ignore_index=True)
    df_avg_acc = df_avg_acc._append(new_row_run_avg_acc_aug_2, ignore_index=True)

    df_avg_acc.to_csv(f"projekt_2_zadanie_2_10_runs_AVG_ACC.csv",   index=False)
    
    new_row_run_std_acc_no_aug = { 
                            'all': std_acc_aug[3], 
                            '100': std_acc_aug[0], 
                            '200': std_acc_aug[1], 
                            '1000': std_acc_aug[2]
    }

    new_row_run_std_acc_aug_1 = {
                            'all': std_acc_aug[7], 
                            '100': std_acc_aug[4], 
                            '200': std_acc_aug[5], 
                            '1000': std_acc_aug[6]
    }
    new_row_run_std_acc_aug_2 = { 
                            'all': std_acc_aug[11], 
                            '100': std_acc_aug[8], 
                            '200': std_acc_aug[9], 
                            '1000': std_acc_aug[10]
    }
    
    
    df_std_div_acc = df_std_div_acc._append(new_row_run_std_acc_no_aug, ignore_index=True)
    df_std_div_acc = df_std_div_acc._append(new_row_run_std_acc_aug_1, ignore_index=True)
    df_std_div_acc = df_std_div_acc._append(new_row_run_std_acc_aug_2, ignore_index=True)

    df_std_div_acc.to_csv(f"projekt_2_zadanie_2_10_runs_STD_ACC.csv",   index=False)
    
        #augmenting_image_ax()


if __name__ == "__main__":
    print('RUNNING FILE RUN TRAINING')
    run_random_state(reduce_dim=False) 
    #run_random_state(reduce_dim=True) 