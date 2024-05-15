import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import torch.nn as nn
from model import CNN_tanh, CNN_sigmoid, CNN_relu, CNN_leaky_relu, CNN_id
import os


def eval_4_models_cifar(data_set, batch_size, data_name, redux):

    lin_out = 128
    if redux:
        lin_out = 20

    model0 = CNN_tanh(in_side_len=32, in_channels=3, cnv0_out_channels=10,
                               cnv1_out_channels=20, lin0_out_size=lin_out, lin1_out_size=10,
                               convolution_kernel=7, pooling_kernel=2, reduce_to_dim2=redux)
    execute_model(data_set, model0, batch_size, data_name+"tanh")

    model3 = CNN_leaky_relu(in_side_len=32, in_channels=3, cnv0_out_channels=10,
                               cnv1_out_channels=20, lin0_out_size=lin_out, lin1_out_size=10,
                               convolution_kernel=7, pooling_kernel=2, reduce_to_dim2=redux)
    execute_model(data_set, model3, batch_size, data_name+"leaky_relu")

    model1 = CNN_relu(in_side_len=32, in_channels=3, cnv0_out_channels=10,
                               cnv1_out_channels=20, lin0_out_size=lin_out, lin1_out_size=10,
                               convolution_kernel=7, pooling_kernel=2, reduce_to_dim2=redux)
    execute_model(data_set, model1, batch_size, data_name+"relu")

    model2 = CNN_sigmoid(in_side_len=32, in_channels=3, cnv0_out_channels=10,
                               cnv1_out_channels=20, lin0_out_size=lin_out, lin1_out_size=10,
                               convolution_kernel=7, pooling_kernel=2, reduce_to_dim2=redux)
    execute_model(data_set, model2, batch_size, data_name+"sigmoid")

    model4 = CNN_id(in_side_len=32, in_channels=3, cnv0_out_channels=10,
                               cnv1_out_channels=20, lin0_out_size=lin_out, lin1_out_size=10,
                               convolution_kernel=7, pooling_kernel=2, reduce_to_dim2=redux)
    execute_model(data_set, model4, batch_size, data_name+"id")


def eval_4_models_mnist(data_set, batch_size, data_name, redux):

    lin_out = 100
    if redux:
        lin_out = 16

    model0 = CNN_tanh(in_side_len=28, in_channels=1, cnv0_out_channels=8,
                                       cnv1_out_channels=16, lin0_out_size=lin_out, lin1_out_size=10,
                                       convolution_kernel=5, pooling_kernel=2, reduce_to_dim2=redux)
    execute_model(data_set, model0, batch_size, data_name+"tanh")

    model3 = CNN_leaky_relu(in_side_len=28, in_channels=1, cnv0_out_channels=8,
                                       cnv1_out_channels=16, lin0_out_size=lin_out, lin1_out_size=10,
                                       convolution_kernel=5, pooling_kernel=2, reduce_to_dim2=redux)
    execute_model(data_set, model3, batch_size, data_name+"leaky_relu")

    model1 = CNN_relu(in_side_len=28, in_channels=1, cnv0_out_channels=8,
                                       cnv1_out_channels=16, lin0_out_size=lin_out, lin1_out_size=10,
                                       convolution_kernel=5, pooling_kernel=2, reduce_to_dim2=redux)
    execute_model(data_set, model1, batch_size, data_name+"relu")

    model2 = CNN_sigmoid(in_side_len=28, in_channels=1, cnv0_out_channels=8,
                                       cnv1_out_channels=16, lin0_out_size=lin_out, lin1_out_size=10,
                                       convolution_kernel=5, pooling_kernel=2, reduce_to_dim2=redux)
    execute_model(data_set, model2, batch_size, data_name+"sigmoid")

    model4 = CNN_id(in_side_len=28, in_channels=1, cnv0_out_channels=8,
                                       cnv1_out_channels=16, lin0_out_size=lin_out, lin1_out_size=10,
                                       convolution_kernel=5, pooling_kernel=2, reduce_to_dim2=redux)
    execute_model(data_set, model4, batch_size, data_name+"id")


def execute_model(data_set, model, batch_size, data_name):
    num_epochs = 50
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'CUDA VERSION: {torch.version.cuda}')
    print(f'DEVICE RUNING: {device}')
    running_loss = []

    criteria = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
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
        running_loss.append([loss.item(), temp_accuracy])

    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", f'model_{data_name}.csv')
    np.savetxt(fname=path, X=running_loss, delimiter=';')

