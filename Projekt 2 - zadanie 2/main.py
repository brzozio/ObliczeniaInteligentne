import exe_model
import eval_model
from datasets_get import cifar10_to_cnn, mnist_to_cnn, cifar10_to_cnn_AUGMENTED
import torch
from model import CNN_tanh, CNN_leaky_relu
import matplotlib.pyplot as plt

train: bool          = False
continue_train: bool = False
batch_size           = 1_000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#data_name = 'projekt_2_zad_2_mnist'
#data_name = 'projekt_2_zad_2_mnist_reduced'
data_name = 'projekt_2_zad_2_cifar10'
#data_name = 'projekt_2_zad_2_cifar10_reduced'

def model_choosing():
    """
            MODELE:
            PIERWSZY - strona 1 i 3
                - mnist         : tanh, lr=0.01, in_side_len=28, in_channels=1, cnv0_out_channels=8, cnv1_out_channels=16, lin0_out_size=100||16, lin1_out_size=10, convolution_kernel=5, pooling_kernel=2, reduce_to_dim2=redux
                - mnist reduced : leaky_relu, lr=0.01, in_side_len=28, in_channels=1, cnv0_out_channels=8, cnv1_out_channels=16, lin0_out_size=100||16, lin1_out_size=10, convolution_kernel=5, pooling_kernel=2, reduce_to_dim2=redux
                - cifar         : tanh, lr=0.001, in_side_len=32, in_channels=3, cnv0_out_channels=10, cnv1_out_channels=20, lin0_out_size=128||20, lin1_out_size=10, convolution_kernel=7, pooling_kernel=2, reduce_to_dim2=redux
                - cifar reduced : tanh, lr=0.001, in_side_len=32, in_channels=3, cnv0_out_channels=10, cnv1_out_channels=20, lin0_out_size=128||20, lin1_out_size=10, convolution_kernel=7, pooling_kernel=2, reduce_to_dim2=redux

            DRUGI - strona 2 i 4
                - mnist         : tanh, lr=0.01, in_side_len=28, in_channels=1, cnv0_out_channels=12, cnv1_out_channels=16, lin0_out_size=100||16, lin1_out_size=10, convolution_kernel=3, pooling_kernel=2, reduce_to_dim2=redux)
                - mnist reduced : leaky_relu, lr=0.01, in_side_len=28, in_channels=1, cnv0_out_channels=4, cnv1_out_channels=16, lin0_out_size=100||16, lin1_out_size=10, convolution_kernel=7, pooling_kernel=2, reduce_to_dim2=redux)
                - cifar         : tanh, lr=0.001, in_side_len=28, in_channels=1, cnv0_out_channels=15, cnv1_out_channels=16, lin0_out_size=100||16, lin1_out_size=10, convolution_kernel=3, pooling_kernel=2, reduce_to_dim2=redux
                - cifar reduced : tanh, lr=0.001, in_side_len=28, in_channels=1, cnv0_out_channels=10, cnv1_out_channels=16, lin0_out_size=100||16, lin1_out_size=10, convolution_kernel=3, pooling_kernel=2, reduce_to_dim2=redux

    """
    mnist      = mnist_to_cnn(device, train)
    #cifar      = cifar10_to_cnn(device, train)

    # Do Accuracy Score
    mnist_train = mnist_to_cnn(device, True)
    cifar_train = cifar10_to_cnn(device, True)

    mnist_test = mnist_to_cnn(device, False)
    cifar_test = cifar10_to_cnn(device, False)

    #model_mnist_ker         = CNN_tanh(in_side_len=28, in_channels=1, cnv0_out_channels=12, cnv1_out_channels=16, lin0_out_size=100, lin1_out_size=10, convolution_kernel=3, pooling_kernel=2, reduce_to_dim2=False)
    #model_mnist_reduced_ker = CNN_leaky_relu(in_side_len=28, in_channels=1, cnv0_out_channels=4, cnv1_out_channels=16, lin0_out_size=16, lin1_out_size=10, convolution_kernel=7, pooling_kernel=2, reduce_to_dim2=True)
    #model_cifar_ker         = CNN_tanh(in_side_len=32, in_channels=3, cnv0_out_channels=15, cnv1_out_channels=16, lin0_out_size=128, lin1_out_size=10, convolution_kernel=7, pooling_kernel=2, reduce_to_dim2=False)
    #model_cifar_reduced_ker = CNN_tanh(in_side_len=32, in_channels=3, cnv0_out_channels=10, cnv1_out_channels=16, lin0_out_size=20, lin1_out_size=10, convolution_kernel=7, pooling_kernel=2, reduce_to_dim2=True)

    #exe_model.execute_model(data_set=mnist, model=model_mnist_activ, batch_size=12_000, data_name='model_mnist_activ', num_epochs=200, lr=0.01, train=train, continue_train=continue_train)
    #exe_model.execute_model(data_set=mnist, model=model_mnist_reduced_activ, batch_size=12_000, data_name='model_mnist_reduced_activ', num_epochs=400, lr=0.01, train=train, continue_train=continue_train)
    #exe_model.execute_model(data_set=mnist, model=model_mnist_ker, batch_size=12_000, data_name='model_mnist_ker', num_epochs=200, lr=0.01, train=train, continue_train=continue_train)
    #exe_model.execute_model(data_set=mnist, model=model_mnist_reduced_ker, batch_size=12_000, data_name='model_mnist_reduced_ker', num_epochs=400, lr=0.01, train=train, continue_train=continue_train)

    #exe_model.execute_model(data_set=cifar, model=model_cifar_activ, batch_size=12_000, data_name='model_cifar_activ', num_epochs=400, lr=0.001, train=train, continue_train=continue_train)
    #exe_model.execute_model(data_set=cifar, model=model_cifar_reduced_activ, batch_size=12_000, data_name='model_cifar_reduced_activ', num_epochs=600, lr=0.001, train=train, continue_train=continue_train)
    #exe_model.execute_model(data_set=cifar, model=model_cifar_ker, batch_size=12_000, data_name='model_cifar_ker', num_epochs=400, lr=0.001, train=train, continue_train=continue_train)
    #exe_model.execute_model(data_set=cifar, model=model_cifar_reduced_ker, batch_size=12_000, data_name='model_cifar_reduced_ker', num_epochs=600, lr=0.001, train=train, continue_train=continue_train)
    """
    model_mnist_activ = CNN_tanh(in_side_len=28, in_channels=1, cnv0_out_channels=8, cnv1_out_channels=16,
                                 lin0_out_size=100, lin1_out_size=10, convolution_kernel=5, pooling_kernel=2,
                                 reduce_to_dim2=False)
    model_mnist_reduced_activ = CNN_leaky_relu(in_side_len=28, in_channels=1, cnv0_out_channels=8, cnv1_out_channels=16,
                                               lin0_out_size=16, lin1_out_size=10, convolution_kernel=5,
                                               pooling_kernel=2, reduce_to_dim2=True)

    exe_model.execute_model_fast(data_set_test=mnist_test, data_set_train=mnist_train, model=model_mnist_activ,
                                 batch_size=12_000, data_name='mnist_activ', num_epoch=100, lr=0.01, calc_interval=2)

    exe_model.execute_model_fast(data_set_test=mnist_test, data_set_train=mnist_train, model=model_mnist_reduced_activ,
                                 batch_size=12_000, data_name='mnist_activ_reduced', num_epoch=200, lr=0.01,
                                 calc_interval=4)
    """

    model_cifar_activ = CNN_tanh(in_side_len=32, in_channels=3, cnv0_out_channels=10, cnv1_out_channels=20,
                                 lin0_out_size=128, lin1_out_size=10, convolution_kernel=7, pooling_kernel=2,
                                 reduce_to_dim2=False)

    model_cifar_reduced_activ = CNN_leaky_relu(in_side_len=32, in_channels=3, cnv0_out_channels=10, cnv1_out_channels=20,
                                         lin0_out_size=20, lin1_out_size=10, convolution_kernel=7, pooling_kernel=2,
                                         reduce_to_dim2=True)

    #exe_model.execute_model_fast(data_set_test=mnist_test, data_set_train=mnist_train, model=model_mnist_ker, batch_size=12_000, data_name='mnist_ker', num_epoch=100, lr=0.01, calc_interval=2)
    #exe_model.execute_model_fast(data_set_test=mnist_test, data_set_train=mnist_train, model=model_mnist_reduced_ker, batch_size=12_000, data_name='mnist_ker_reduced', num_epoch=200, lr=0.01, calc_interval=4)

    #exe_model.execute_model_fast(data_set_test=cifar_test, data_set_train=cifar_train, model=model_cifar_activ,
    #                             batch_size=5000, data_name='cifar_ker', num_epoch=400, lr=0.001, calc_interval=8)

    exe_model.execute_model_fast(data_set_test=cifar_test, data_set_train=cifar_train, model=model_cifar_reduced_activ,
                                 batch_size=5000, data_name='cifar_reduced_ker', num_epoch=600, lr=0.005,
                                 calc_interval=12)


if __name__ == "__main__":
    match data_name:
        case 'projekt_2_zad_2_mnist':
            model = exe_model.CNN_tanh(in_side_len=28, in_channels=1, cnv0_out_channels=8,
                                  cnv1_out_channels=16, lin0_out_size=20, lin1_out_size=10,
                                  convolution_kernel=5, pooling_kernel=2)
            data_set      = mnist_to_cnn(device, train)
            

        case 'projekt_2_zad_2_mnist_reduced':
            model = exe_model.CNN_tanh(in_side_len=28, in_channels=1, cnv0_out_channels=8,
                                  reduce_to_dim2=True, lin0_out_size=20, lin1_out_size=10,
                                  convolution_kernel=5, pooling_kernel=2)
            data_set      = mnist_to_cnn(device, train)

        case 'projekt_2_zad_2_cifar10':
            model = exe_model.CNN_tanh(in_side_len=32, in_channels=3, cnv0_out_channels=6,
                                  cnv1_out_channels=16, lin0_out_size=128, lin1_out_size=10,
                                  convolution_kernel=3, pooling_kernel=2)
            #data_set      = cifar10_to_cnn_AUGMENTED(device, train)
            data_set      = cifar10_to_cnn(device, train)

        case 'projekt_2_zad_2_cifar10_reduced':
            model = exe_model.CNN_tanh(in_side_len=32, in_channels=3, cnv0_out_channels=10,
                                  reduce_to_dim2=True, lin0_out_size=128, lin1_out_size=10,
                                  convolution_kernel=3, pooling_kernel=2)
            data_set      = cifar10_to_cnn(device, train)

        case _:
            model         = None
            data_set      = None

    if (model is not None) or (data_set is not None):
        """
        exe_model.execute_model(data_set=data_set, model=model, batch_size=batch_size, data_name=data_name,
                                train=train, continue_train=continue_train)
        """
        #eval_model.eval_4_models_cifar(data_set=data_set, batch_size=batch_size, data_name=data_name, redux=False)

        #eval_model.eval_4_models_mnist(data_set=data_set, batch_size=batch_size, data_name=data_name, redux=False)

        #eval_model.eval_models_mnist_sizes(data_set=data_set, batch_size=batch_size, data_name=data_name)

        #eval_model.eval_models_cifar_sizes(data_set=data_set, batch_size=batch_size, data_name=data_name)
   

    model_choosing()


