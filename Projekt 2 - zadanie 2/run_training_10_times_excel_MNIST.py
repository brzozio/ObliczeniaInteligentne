import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier as MLP
from voronoi import plot_decision_boundary, plot_decision_boundary_ax
from sklearn.metrics import accuracy_score
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import torch
from model import CNN_tanh, CNN_leaky_relu
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
])

rotate = transforms.Compose([
    #Rotacja obrazu o rand kąt <0,15> 
    transforms.RandomRotation(30),          
])

color_jitter = transforms.Compose([
    #Modulacja jasności, kontrastu, nasycenia w obrazie - sens, zdjęcia mogą być ciemniejsze itd.
    transforms.ColorJitter(brightness=0.7, contrast=0.8, saturation=0.4, hue=0.3),  
])

random_crop = transforms.Compose([
    #Randomowe przycięcie obrazu - moze np. wyciąć samą głowę zwierzęcia
    transforms.RandomCrop(32, padding=4),                   
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def augmenting_image_ax(transform, fname=None):
    mnist = datasets.MNIST(
            root='data',
            train=True,
            download=True,
    )
    _, ax = plt.subplots(3, 2, figsize=(10, 20))

    for row in range(3):
        for col in range(2):
            index = (row * 2) + col
            original_image, _ = mnist[index]
            augmented_image = transform(original_image)
            ax[row, 0].imshow(original_image, cmap='gray')
            ax[row, 0].set_title(f'Original Image')
            ax[row, 1].imshow(augmented_image, cmap='gray')
            ax[row, 1].set_title(f'Augmented Image')

    if fname is None:
        plt.show()
    else:
        plt.savefig(fname)

    # a w przypadku 2 cech należy zaprezentować rozkład danych treningowych w przypadku gdy
    # rozważane było 100 przykładów raz gdy tylko te dane są widoczne i
    # raz gdy dla każdej danej 10 razy została zastosowana metoda augmentacji (1000 przykładów).

def collate_fn(batch):
    # Konwertuj obrazy PIL na tensory
    images, labels = zip(*batch)
    images = torch.stack([transforms.ToTensor()(img) for img in images])
    return images, torch.tensor(labels)

def visualize_data_distribution(transform=None, fname=None):
    mnist = datasets.MNIST(
        root='data',
        train=True,
        download=True,
    )
    data_loader = DataLoader(mnist, batch_size=100, shuffle=True, collate_fn=collate_fn)
    images, labels = next(iter(data_loader))

    plt.figure(figsize=(12, 6))

    # Histogram przed transformacją
    plt.subplot(1, 2, 1)
    plt.hist(labels.numpy(), bins=range(11), edgecolor='black')
    plt.title('Data Distribution (100 examples)')
    plt.xlabel('Class')
    plt.ylabel('Frequency')

    if transform:
        augmented_images = []
        augmented_labels = []
        for i in range(10):
            augmented_images.extend(transform(image) for image in images)
            augmented_labels.extend(labels)
        augmented_labels = torch.tensor(augmented_labels)
        plt.subplot(1, 2, 2)
        plt.hist(augmented_labels.numpy(), bins=range(11), edgecolor='black')
        plt.title('Data Distribution (1000 examples with augmentation)')
        plt.xlabel('Class')
        plt.ylabel('Frequency')

    if fname is None:
        plt.show()
    else:
        plt.savefig(fname)



def mnist_to_cnn(device, train, transforming) -> CustomDataset:
    mnist           = datasets.MNIST(root='data', train=train, download=True, transform=transforming)
    mnists          = CustomDataset(data=mnist.data, targets=mnist.targets, device=device)
    mnists.data     = mnists.data.view(-1,1,28,28)
    return mnists

def run_random_state(model, num_runs) -> None:
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
    
    
    save_model(model.state_dict(), f'RUN_10_TIMES____START_DICT_MNIST.pth') 
    
    augmentations = [basic, rotate, color_jitter]
    sample_sizes  = [100,200,1000,60000]
    criteria      = torch.nn.CrossEntropyLoss()
    num_epochs    = 50
    
    avg_acc_aug           = np.array([])
    std_acc_aug           = np.array([])

    data_set_basic_test   = mnist_to_cnn(device, False, basic)

    for augm_i, augm in enumerate(augmentations):
        data_set_train        = mnist_to_cnn(device, True, augm)

        for sample_size in sample_sizes: 
            print(f"SAMPLE SIZE: {sample_size}")
            accuracy_score_list = np.array([])
            max_accuracy        = 0
               
            if sample_size in (100,200):
                batch_size   = 10
            elif sample_size == 1_000:
                batch_size   = 100
            else:
                batch_size = 5000

            print(f'BATCH SIZE FOR {augm_i} AUG {sample_size} is: {batch_size}')
            
            sample_param = torch.randperm(len(data_set_train))[:sample_size]
            sampler      = SubsetRandomSampler(sample_param)
            dataloader   = DataLoader(dataset=data_set_train, batch_size=batch_size, sampler=sampler,drop_last=True)

            for run in range(num_runs):
                model.load_state_dict(load_model(f'RUN_10_TIMES____START_DICT_MNIST.pth'))

                optimizer     = torch.optim.Adam(model.parameters(), lr=0.01)
                model.train()
                model.double()
                model.to(device)

                for epoch in range(num_epochs):
                    for batch in dataloader:
                        data, target = batch['data'].to(device), batch['target'].to(device)
                        outputs = model.extract(data)
                        outputs = model.forward(outputs)
                        loss = criteria(outputs, target)
                        
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    
                    print(f"Epoch [{epoch+1}/{num_epochs}]  Loss: {loss.item():.5f} - AUGM {augm_i}, SAMPLE {sample_size}, RUN [{run}]")

                model.eval()

                outputs = model.extract(data_set_basic_test.data)
                outputs = model.forward(outputs)
                
                softmax           = torch.nn.Softmax(dim=1)
                probabilities     = softmax(outputs)
                predicted_classes = torch.argmax(probabilities, dim=1)

                predicted_classes_cpu = predicted_classes.cpu().numpy()
                targets_cpu           = data_set_basic_test.targets.cpu().numpy()

                accuracy            = accuracy_score(predicted_classes_cpu, targets_cpu)
                accuracy_score_list = np.append(accuracy_score_list,accuracy)
                print(f'ACCURACY SCORE: {accuracy:.4f}')
                
                if accuracy > max_accuracy:
                    max_accuracy = accuracy
                    model.train()
                    save_model(model.state_dict(), f'RUN_10_TIMES_{augm_i}_{sample_size}_red_{model.reduce_to_dim2}_MNIST.pth') 
                    model.eval()

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

    df_avg_acc.to_csv(f"projekt_2_zadanie_2_10_runs_AVG_ACC_red_{model.reduce_to_dim2}_MNIST.csv", index=False)
    
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

    df_std_div_acc.to_csv(f"projekt_2_zadanie_2_10_runs_STD_ACC_red_{model.reduce_to_dim2}_MNIST.csv", index=False)
    
    if model.reduce_to_dim2 is True:
        augmenting_image_ax(transform=color_jitter)
        augmenting_image_ax(transform=rotate)
        visualize_data_distribution(transform=None)
        visualize_data_distribution(transform=color_jitter)


if __name__ == "__main__":

    model_mnist_activ         = CNN_tanh(in_side_len=28, in_channels=1, cnv0_out_channels=8, cnv1_out_channels=16, lin0_out_size=100, lin1_out_size=10, convolution_kernel=5, pooling_kernel=2, reduce_to_dim2=False)
    model_mnist_reduced_activ = CNN_leaky_relu(in_side_len=28, in_channels=1, cnv0_out_channels=8, cnv1_out_channels=16, lin0_out_size=16, lin1_out_size=10, convolution_kernel=5, pooling_kernel=2, reduce_to_dim2=True)
    model_mnist_ker         = CNN_tanh(in_side_len=28, in_channels=1, cnv0_out_channels=12, cnv1_out_channels=16, lin0_out_size=100, lin1_out_size=10, convolution_kernel=3, pooling_kernel=2, reduce_to_dim2=False)
    model_mnist_reduced_ker = CNN_leaky_relu(in_side_len=28, in_channels=1, cnv0_out_channels=4, cnv1_out_channels=16, lin0_out_size=16, lin1_out_size=10, convolution_kernel=7, pooling_kernel=2, reduce_to_dim2=True)
    


    print('RUNNING FILE RUN TRAINING')
    #run_random_state(model=model_mnist_ker, num_runs=10)
    #run_random_state(model=model_mnist_reduced_ker, num_runs=10) \

    augmenting_image_ax(transforms.ColorJitter(brightness=2, contrast=3, saturation=0.5, hue=0.5))
    augmenting_image_ax(transforms.RandomRotation(15))