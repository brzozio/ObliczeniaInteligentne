import exe_model

if __name__ == "__main__":

    model = exe_model.CNN(in_side_len=28, in_channels=1, cnv0_out_channels=8,
                reduce_to_dim2=True, lin0_out_size=20, lin1_out_size=10,
                convolution_kernel=5, pooling_kernel=2)

    data_name = 'projekt_2_zad_2_mnist'
    model_path = f'model_{data_name}_reduce.pth'

    """
    model = CNN(in_side_len=32, in_channels=3, cnv0_out_channels=6,
                cnv1_out_channels=16, lin0_out_size=120, lin1_out_size=10,
                lin2_out_size=0, convolution_kernel=3, pooling_kernel=2)
    
    data_name = 'projekt_2_zad_2_cifar10'
    model_path = f'model_{data_name}.pth'
    """

    exe_model.execute_model(model, data_name, model_path)