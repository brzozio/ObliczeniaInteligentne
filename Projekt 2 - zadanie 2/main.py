import exe_model
from datasets_get import cifar10_to_cnn, mnist_to_cnn, cifar10_to_cnn_AUGMENTED
import torch

train: bool          = False
continue_train: bool = False
batch_size           = 20_000
#data_name = 'projekt_2_zad_2_mnist'
#data_name = 'projekt_2_zad_2_mnist_reduced'
data_name = 'projekt_2_zad_2_cifar10'
#data_name = 'projekt_2_zad_2_cifar10_reduced'

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    match data_name:
        case 'projekt_2_zad_2_mnist':
            model = exe_model.CNN(in_side_len=28, in_channels=1, cnv0_out_channels=8,
                                  cnv1_out_channels=16, lin0_out_size=20, lin1_out_size=10,
                                  convolution_kernel=5, pooling_kernel=2)
            data_set      = mnist_to_cnn(device, train)

        case 'projekt_2_zad_2_mnist_reduced':
            model = exe_model.CNN(in_side_len=28, in_channels=1, cnv0_out_channels=8,
                                  reduce_to_dim2=True, lin0_out_size=20, lin1_out_size=10,
                                  convolution_kernel=5, pooling_kernel=2)
            data_set      = mnist_to_cnn(device, train)

        case 'projekt_2_zad_2_cifar10':
            model = exe_model.CNN(in_side_len=32, in_channels=3, cnv0_out_channels=6,
                                  cnv1_out_channels=16, lin0_out_size=128, lin1_out_size=10,
                                  convolution_kernel=3, pooling_kernel=2)
            #data_set      = cifar10_to_cnn_AUGMENTED(device, train)
            data_set      = cifar10_to_cnn(device, train)

        case 'projekt_2_zad_2_cifar10_reduced':
            model = exe_model.CNN(in_side_len=32, in_channels=3, cnv0_out_channels=10,
                                  reduce_to_dim2=True, lin0_out_size=128, lin1_out_size=10,
                                  convolution_kernel=3, pooling_kernel=2)
            data_set      = cifar10_to_cnn(device, train)

        case _:
            model         = None
            data_set      = None

    if (model is not None) or (data_set is not None):
        exe_model.execute_model(data_set=data_set, model=model, batch_size=batch_size, data_name=data_name,
                                train=train, continue_train=continue_train)
