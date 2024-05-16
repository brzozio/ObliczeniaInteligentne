import joblib
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import torch.nn as nn
from model import CNN_tanh, CNN_sigmoid, CNN_relu, CNN_leaky_relu, CNN_id
from joblib import dump

def eval_4_models_cifar(data_set, batch_size, data_name, redux):

    acc_ar = np.zeros((5,50))

    lin_out = 128
    if redux:
        lin_out = 20

    model0 = CNN_tanh(in_side_len=32, in_channels=3, cnv0_out_channels=10,
                               cnv1_out_channels=4, lin0_out_size=lin_out, lin1_out_size=10,
                               convolution_kernel=7, pooling_kernel=2, reduce_to_dim2=redux)
    acc_ar[0] = eval_model(data_set, model0, batch_size, data_name+"tanh_0_01", learning_rate=0.001)
    """
    model1 = CNN_tanh(in_side_len=32, in_channels=3, cnv0_out_channels=10,
                               cnv1_out_channels=20, lin0_out_size=lin_out, lin1_out_size=10,
                               convolution_kernel=7, pooling_kernel=2, reduce_to_dim2=redux)
    acc_ar[1] = eval_model(data_set, model1, batch_size, data_name+"tanh_0_001", learning_rate=0.001)

    
    model3 = CNN_leaky_relu(in_side_len=32, in_channels=3, cnv0_out_channels=10,
                               cnv1_out_channels=20, lin0_out_size=lin_out, lin1_out_size=10,
                               convolution_kernel=7, pooling_kernel=2, reduce_to_dim2=redux)
    acc_ar[2] = eval_model(data_set, model3, batch_size, data_name+"leaky_relu")

    model1 = CNN_relu(in_side_len=32, in_channels=3, cnv0_out_channels=10,
                               cnv1_out_channels=20, lin0_out_size=lin_out, lin1_out_size=10,
                               convolution_kernel=7, pooling_kernel=2, reduce_to_dim2=redux)
    acc_ar[1] = eval_model(data_set, model1, batch_size, data_name+"relu")

    model2 = CNN_sigmoid(in_side_len=32, in_channels=3, cnv0_out_channels=10,
                               cnv1_out_channels=20, lin0_out_size=lin_out, lin1_out_size=10,
                               convolution_kernel=7, pooling_kernel=2, reduce_to_dim2=redux)
    acc_ar[3] = eval_model(data_set, model2, batch_size, data_name+"sigmoid")

    model4 = CNN_id(in_side_len=32, in_channels=3, cnv0_out_channels=10,
                               cnv1_out_channels=20, lin0_out_size=lin_out, lin1_out_size=10,
                               convolution_kernel=7, pooling_kernel=2, reduce_to_dim2=redux)
    acc_ar[0] = eval_model(data_set, model4, batch_size, data_name+"id")
    """
    #print(acc_ar)

def eval_4_models_mnist(data_set, batch_size, data_name, redux):

    acc_ar = np.zeros((5,50))

    lin_out = 100
    if redux:
        lin_out = 16

    model0 = CNN_tanh(in_side_len=28, in_channels=1, cnv0_out_channels=8,
                                       cnv1_out_channels=16, lin0_out_size=lin_out, lin1_out_size=10,
                                       convolution_kernel=5, pooling_kernel=2, reduce_to_dim2=redux)
    acc_ar[4] = eval_model(data_set, model0, batch_size, data_name+"tanh")

    model3 = CNN_leaky_relu(in_side_len=28, in_channels=1, cnv0_out_channels=8,
                                       cnv1_out_channels=16, lin0_out_size=lin_out, lin1_out_size=10,
                                       convolution_kernel=5, pooling_kernel=2, reduce_to_dim2=redux)
    acc_ar[2] = eval_model(data_set, model3, batch_size, data_name+"leaky_relu")

    model1 = CNN_relu(in_side_len=28, in_channels=1, cnv0_out_channels=8,
                                       cnv1_out_channels=16, lin0_out_size=lin_out, lin1_out_size=10,
                                       convolution_kernel=5, pooling_kernel=2, reduce_to_dim2=redux)
    acc_ar[1] = eval_model(data_set, model1, batch_size, data_name+"relu")

    model2 = CNN_sigmoid(in_side_len=28, in_channels=1, cnv0_out_channels=8,
                                       cnv1_out_channels=16, lin0_out_size=lin_out, lin1_out_size=10,
                                       convolution_kernel=5, pooling_kernel=2, reduce_to_dim2=redux)
    acc_ar[3] = eval_model(data_set, model2, batch_size, data_name+"sigmoid")

    model4 = CNN_id(in_side_len=28, in_channels=1, cnv0_out_channels=8,
                                       cnv1_out_channels=16, lin0_out_size=lin_out, lin1_out_size=10,
                                       convolution_kernel=5, pooling_kernel=2, reduce_to_dim2=redux)
    acc_ar[0] = eval_model(data_set, model4, batch_size, data_name+"id")


def eval_models_mnist_sizes(data_set, batch_size, data_name):

    acc_ar = np.zeros((2, 9, 50))

    joblib.dump(acc_ar, 'mnist_kernel_channels.joblib')

    model0 = CNN_tanh(in_side_len=28, in_channels=1, cnv0_out_channels=4,
                                       cnv1_out_channels=16, lin0_out_size=100, lin1_out_size=10,
                                       convolution_kernel=3, pooling_kernel=2, reduce_to_dim2=False)
    acc_ar[0][0] = eval_model(data_set, model0, batch_size, data_name, learning_rate=0.01)

    model1 = CNN_tanh(in_side_len=28, in_channels=1, cnv0_out_channels=8,
                                       cnv1_out_channels=16, lin0_out_size=100, lin1_out_size=10,
                                       convolution_kernel=3, pooling_kernel=2, reduce_to_dim2=False)
    acc_ar[0][1] = eval_model(data_set, model1, batch_size, data_name, learning_rate=0.01)

    model2 = CNN_tanh(in_side_len=28, in_channels=1, cnv0_out_channels=12,
                                       cnv1_out_channels=16, lin0_out_size=100, lin1_out_size=10,
                                       convolution_kernel=3, pooling_kernel=2, reduce_to_dim2=False)
    acc_ar[0][2] = eval_model(data_set, model2, batch_size, data_name, learning_rate=0.01)

    model3 = CNN_tanh(in_side_len=28, in_channels=1, cnv0_out_channels=4,
                                       cnv1_out_channels=16, lin0_out_size=100, lin1_out_size=10,
                                       convolution_kernel=5, pooling_kernel=2, reduce_to_dim2=False)
    acc_ar[0][3] = eval_model(data_set, model3, batch_size, data_name, learning_rate=0.01)

    model4 = CNN_tanh(in_side_len=28, in_channels=1, cnv0_out_channels=8,
                                       cnv1_out_channels=16, lin0_out_size=100, lin1_out_size=10,
                                       convolution_kernel=5, pooling_kernel=2, reduce_to_dim2=False)
    acc_ar[0][4] = eval_model(data_set, model4, batch_size, data_name, learning_rate=0.01)

    model5 = CNN_tanh(in_side_len=28, in_channels=1, cnv0_out_channels=12,
                                       cnv1_out_channels=16, lin0_out_size=100, lin1_out_size=10,
                                       convolution_kernel=5, pooling_kernel=2, reduce_to_dim2=False)
    acc_ar[0][5] = eval_model(data_set, model5, batch_size, data_name, learning_rate=0.01)

    model6 = CNN_tanh(in_side_len=28, in_channels=1, cnv0_out_channels=4,
                                       cnv1_out_channels=16, lin0_out_size=100, lin1_out_size=10,
                                       convolution_kernel=7, pooling_kernel=2, reduce_to_dim2=False)
    acc_ar[0][6] = eval_model(data_set, model6, batch_size, data_name, learning_rate=0.01)

    model7 = CNN_tanh(in_side_len=28, in_channels=1, cnv0_out_channels=8,
                                       cnv1_out_channels=16, lin0_out_size=100, lin1_out_size=10,
                                       convolution_kernel=7, pooling_kernel=2, reduce_to_dim2=False)
    acc_ar[0][7] = eval_model(data_set, model7, batch_size, data_name, learning_rate=0.01)

    model8 = CNN_tanh(in_side_len=28, in_channels=1, cnv0_out_channels=12,
                                       cnv1_out_channels=16, lin0_out_size=100, lin1_out_size=10,
                                       convolution_kernel=7, pooling_kernel=2, reduce_to_dim2=False)
    acc_ar[0][8] = eval_model(data_set, model8, batch_size, data_name, learning_rate=0.01)

    ############################################################################################

    model10 = CNN_leaky_relu(in_side_len=28, in_channels=1, cnv0_out_channels=4,
                                       cnv1_out_channels=16, lin0_out_size=16, lin1_out_size=10,
                                       convolution_kernel=3, pooling_kernel=2, reduce_to_dim2=True)
    acc_ar[1][0] = eval_model(data_set, model10, batch_size, data_name, learning_rate=0.01)

    model11 = CNN_leaky_relu(in_side_len=28, in_channels=1, cnv0_out_channels=8,
                                       cnv1_out_channels=16, lin0_out_size=16, lin1_out_size=10,
                                       convolution_kernel=3, pooling_kernel=2, reduce_to_dim2=True)
    acc_ar[1][1] = eval_model(data_set, model11, batch_size, data_name, learning_rate=0.01)

    model12 = CNN_leaky_relu(in_side_len=28, in_channels=1, cnv0_out_channels=12,
                                       cnv1_out_channels=16, lin0_out_size=16, lin1_out_size=10,
                                       convolution_kernel=3, pooling_kernel=2, reduce_to_dim2=True)
    acc_ar[1][2] = eval_model(data_set, model12, batch_size, data_name, learning_rate=0.01)

    model13 = CNN_leaky_relu(in_side_len=28, in_channels=1, cnv0_out_channels=4,
                                       cnv1_out_channels=16, lin0_out_size=16, lin1_out_size=10,
                                       convolution_kernel=5, pooling_kernel=2, reduce_to_dim2=True)
    acc_ar[1][3] = eval_model(data_set, model13, batch_size, data_name, learning_rate=0.01)

    model14 = CNN_leaky_relu(in_side_len=28, in_channels=1, cnv0_out_channels=8,
                                       cnv1_out_channels=16, lin0_out_size=16, lin1_out_size=10,
                                       convolution_kernel=5, pooling_kernel=2, reduce_to_dim2=True)
    acc_ar[1][4] = eval_model(data_set, model14, batch_size, data_name, learning_rate=0.01)

    model15 = CNN_leaky_relu(in_side_len=28, in_channels=1, cnv0_out_channels=12,
                                       cnv1_out_channels=16, lin0_out_size=16, lin1_out_size=10,
                                       convolution_kernel=5, pooling_kernel=2, reduce_to_dim2=True)
    acc_ar[1][5] = eval_model(data_set, model15, batch_size, data_name, learning_rate=0.01)

    model16 = CNN_leaky_relu(in_side_len=28, in_channels=1, cnv0_out_channels=4,
                                       cnv1_out_channels=16, lin0_out_size=16, lin1_out_size=10,
                                       convolution_kernel=7, pooling_kernel=2, reduce_to_dim2=True)
    acc_ar[1][6] = eval_model(data_set, model16, batch_size, data_name, learning_rate=0.01)

    model17 = CNN_leaky_relu(in_side_len=28, in_channels=1, cnv0_out_channels=8,
                                       cnv1_out_channels=16, lin0_out_size=16, lin1_out_size=10,
                                       convolution_kernel=7, pooling_kernel=2, reduce_to_dim2=True)
    acc_ar[1][7] = eval_model(data_set, model17, batch_size, data_name, learning_rate=0.01)

    model18 = CNN_leaky_relu(in_side_len=28, in_channels=1, cnv0_out_channels=12,
                                       cnv1_out_channels=16, lin0_out_size=16, lin1_out_size=10,
                                       convolution_kernel=7, pooling_kernel=2, reduce_to_dim2=True)
    acc_ar[1][8] = eval_model(data_set, model18, batch_size, data_name, learning_rate=0.01)

    joblib.dump(acc_ar, 'mnist_kernel_channels.joblib')


def eval_models_cifar_sizes(data_set, batch_size, data_name):

    acc_ar = np.zeros((2, 9, 50))

    joblib.dump(acc_ar, 'cifar_kernel_channels.joblib')

    model0 = CNN_tanh(in_side_len=32, in_channels=3, cnv0_out_channels=5,
                      cnv1_out_channels=20, lin0_out_size=128, lin1_out_size=10,
                      convolution_kernel=3, pooling_kernel=2, reduce_to_dim2=False)
    acc_ar[0][0] = eval_model(data_set, model0, batch_size, data_name, learning_rate=0.001)

    model1 = CNN_tanh(in_side_len=32, in_channels=3, cnv0_out_channels=10,
                      cnv1_out_channels=20, lin0_out_size=128, lin1_out_size=10,
                      convolution_kernel=3, pooling_kernel=2, reduce_to_dim2=False)
    acc_ar[0][1] = eval_model(data_set, model1, batch_size, data_name, learning_rate=0.001)

    model2 = CNN_tanh(in_side_len=32, in_channels=3, cnv0_out_channels=15,
                      cnv1_out_channels=20, lin0_out_size=128, lin1_out_size=10,
                      convolution_kernel=3, pooling_kernel=2, reduce_to_dim2=False)
    acc_ar[0][2] = eval_model(data_set, model2, batch_size, data_name, learning_rate=0.001)

    model3 = CNN_tanh(in_side_len=32, in_channels=3, cnv0_out_channels=5,
                      cnv1_out_channels=20, lin0_out_size=128, lin1_out_size=10,
                      convolution_kernel=5, pooling_kernel=2, reduce_to_dim2=False)
    acc_ar[0][3] = eval_model(data_set, model3, batch_size, data_name, learning_rate=0.001)

    model4 = CNN_tanh(in_side_len=32, in_channels=3, cnv0_out_channels=10,
                      cnv1_out_channels=20, lin0_out_size=128, lin1_out_size=10,
                      convolution_kernel=5, pooling_kernel=2, reduce_to_dim2=False)
    acc_ar[0][4] = eval_model(data_set, model4, batch_size, data_name, learning_rate=0.001)

    model5 = CNN_tanh(in_side_len=32, in_channels=3, cnv0_out_channels=15,
                      cnv1_out_channels=20, lin0_out_size=128, lin1_out_size=10,
                      convolution_kernel=5, pooling_kernel=2, reduce_to_dim2=False)
    acc_ar[0][5] = eval_model(data_set, model5, batch_size, data_name, learning_rate=0.001)

    model6 = CNN_tanh(in_side_len=32, in_channels=3, cnv0_out_channels=5,
                      cnv1_out_channels=20, lin0_out_size=128, lin1_out_size=10,
                      convolution_kernel=7, pooling_kernel=2, reduce_to_dim2=False)
    acc_ar[0][6] = eval_model(data_set, model6, batch_size, data_name, learning_rate=0.001)

    model7 = CNN_tanh(in_side_len=32, in_channels=3, cnv0_out_channels=10,
                      cnv1_out_channels=20, lin0_out_size=128, lin1_out_size=10,
                      convolution_kernel=7, pooling_kernel=2, reduce_to_dim2=False)
    acc_ar[0][7] = eval_model(data_set, model7, batch_size, data_name, learning_rate=0.001)

    model8 = CNN_tanh(in_side_len=32, in_channels=3, cnv0_out_channels=15,
                      cnv1_out_channels=20, lin0_out_size=128, lin1_out_size=10,
                      convolution_kernel=7, pooling_kernel=2, reduce_to_dim2=False)
    acc_ar[0][8] = eval_model(data_set, model8, batch_size, data_name, learning_rate=0.001)

    ############################################################################################

    model10 = CNN_tanh(in_side_len=32, in_channels=3, cnv0_out_channels=5,
                      cnv1_out_channels=20, lin0_out_size=20, lin1_out_size=10,
                      convolution_kernel=3, pooling_kernel=2, reduce_to_dim2=True)
    acc_ar[1][0] = eval_model(data_set, model10, batch_size, data_name, learning_rate=0.001)

    model11 = CNN_tanh(in_side_len=32, in_channels=3, cnv0_out_channels=10,
                      cnv1_out_channels=20, lin0_out_size=20, lin1_out_size=10,
                      convolution_kernel=3, pooling_kernel=2, reduce_to_dim2=True)
    acc_ar[1][1] = eval_model(data_set, model11, batch_size, data_name, learning_rate=0.001)

    model12 = CNN_tanh(in_side_len=32, in_channels=3, cnv0_out_channels=15,
                      cnv1_out_channels=20, lin0_out_size=20, lin1_out_size=10,
                      convolution_kernel=3, pooling_kernel=2, reduce_to_dim2=True)
    acc_ar[1][2] = eval_model(data_set, model12, batch_size, data_name, learning_rate=0.001)

    model13 = CNN_tanh(in_side_len=32, in_channels=3, cnv0_out_channels=5,
                      cnv1_out_channels=20, lin0_out_size=20, lin1_out_size=10,
                      convolution_kernel=5, pooling_kernel=2, reduce_to_dim2=True)
    acc_ar[1][3] = eval_model(data_set, model13, batch_size, data_name, learning_rate=0.001)

    model14 = CNN_tanh(in_side_len=32, in_channels=3, cnv0_out_channels=10,
                      cnv1_out_channels=20, lin0_out_size=20, lin1_out_size=10,
                      convolution_kernel=5, pooling_kernel=2, reduce_to_dim2=True)
    acc_ar[1][4] = eval_model(data_set, model14, batch_size, data_name, learning_rate=0.001)

    model15 = CNN_tanh(in_side_len=32, in_channels=3, cnv0_out_channels=15,
                      cnv1_out_channels=20, lin0_out_size=20, lin1_out_size=10,
                      convolution_kernel=5, pooling_kernel=2, reduce_to_dim2=True)
    acc_ar[1][5] = eval_model(data_set, model15, batch_size, data_name, learning_rate=0.001)

    model16 = CNN_tanh(in_side_len=32, in_channels=3, cnv0_out_channels=5,
                      cnv1_out_channels=20, lin0_out_size=20, lin1_out_size=10,
                      convolution_kernel=7, pooling_kernel=2, reduce_to_dim2=True)
    acc_ar[1][6] = eval_model(data_set, model16, batch_size, data_name, learning_rate=0.001)

    model17 = CNN_tanh(in_side_len=32, in_channels=3, cnv0_out_channels=10,
                      cnv1_out_channels=20, lin0_out_size=20, lin1_out_size=10,
                      convolution_kernel=7, pooling_kernel=2, reduce_to_dim2=True)
    acc_ar[1][7] = eval_model(data_set, model17, batch_size, data_name, learning_rate=0.001)

    model18 = CNN_tanh(in_side_len=32, in_channels=3, cnv0_out_channels=15,
                      cnv1_out_channels=20, lin0_out_size=20, lin1_out_size=10,
                      convolution_kernel=7, pooling_kernel=2, reduce_to_dim2=True)
    acc_ar[1][8] = eval_model(data_set, model18, batch_size, data_name, learning_rate=0.001)

    joblib.dump(acc_ar, 'cifar_kernel_channels.joblib')



def eval_model(data_set, model, batch_size, data_name, learning_rate) -> np.ndarray:

    num_epochs = 50
    running_acc = np.zeros(50)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'CUDA VERSION: {torch.version.cuda}')
    print(f'DEVICE RUNING: {device}')

    criteria = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    model.double()
    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=False)
    print(f'DATA SIZE: {data_set.data.size()}')

    model.to(device)

    for epoch in range(num_epochs):
        for batch in data_loader:
            data, target = batch['data'].to(device), batch['target'].to(device)
            outputs = model.extract(data)
            outputs = model.forward(outputs)
            loss = criteria(outputs, target)

            optimizer.zero_grad()  # Zerowanie gradientów, aby git auniknąć akumulacji w kolejnych krokach
            loss.backward()  # Backpropagation: Obliczenie gradientów
            optimizer.step()  # Aktualizacja wag

        print(f"Epoch [{epoch + 1}/{num_epochs}]  Loss: {loss.item():.5f}   - {data_name}")
        model.eval()
        outputs = model.forward(model.extract(data_set.data))
        predicted_classes = torch.argmax(outputs, dim=1).cpu().numpy()
        temp_accuracy = accuracy_score(predicted_classes, data_set.targets.cpu().numpy())
        running_acc[epoch] = temp_accuracy

    #path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", f'model_{data_name}.csv')
    #np.savetxt(fname=path, X=running_acc, delimiter=';')
    return running_acc


