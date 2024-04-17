import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier as MLP
from voronoi import plot_decision_boundary, plot_decision_boundary_ax
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from joblib import dump as model_dump
from joblib import load as model_load

hidden_neurons = [2,10]

train_accuracies_all = []
test_accuracies_all  = []



def train_and_evaluate(X_train, X_test, y_train, y_test, hidden_neurons, exp, random_state=None) -> tuple[list, list, float, int, float, float]:
    model = MLP(hidden_layer_sizes=hidden_neurons[exp], max_iter=100000, random_state=random_state, solver='sgd', tol=0, n_iter_no_change=100000, activation='identity')
    train_accuracies = []
    test_accuracies  = []
    best_epoch_num : int   = 0
    best_epoch_acc : float = 0.0

    for epoch in range(100000):  
        model.partial_fit(X_train, y_train, classes=np.unique(y_train))
        train_accuracy = accuracy_score(y_train, model.predict(X_train))
        test_accuracy  = accuracy_score(y_test, model.predict(X_test))
        if epoch is 0: 
            model_dump(model, f'mlp_moodel_exp_{exp+2}_EPOCH_0.joblib')
            start_acc_test  = test_accuracy
            start_acc_train = train_accuracy
        if test_accuracy > best_epoch_acc:
            best_epoch_acc = test_accuracy
            best_epoch_num = epoch
            model_dump(model, f'mlp_model_exp_{exp+2}_BEST.joblib')
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)
        print(f"Epoch: {epoch}, Test Accuracy={test_accuracy}, Train Accuracy={train_accuracy}")
    model_dump(model, f'mlp_moodel_exp_{exp+2}_EPOCH_LAST.joblib')
    end_acc_test  = test_accuracy
    end_acc_train = train_accuracy

    return train_accuracies, test_accuracies, best_epoch_acc, best_epoch_num, start_acc_test, start_acc_train, end_acc_test, end_acc_train

def run_random_state(num_runs, Data_train, experiment)->None:
    df_test = pd.DataFrame({
        'Run': [],
        'AccStart': [],
        'AccBest': [],
        'AccBestEpochNr': [],
        'AccEnd': []
    })
    df_train = pd.DataFrame({
        'Run': [],
        'AccStart': [],
        'AccBest': [],
        'AccBestEpochNr': [],
        'AccEnd': []
    })
    for run in range(num_runs):
        train_accuracies, test_accuracies, best_epoch_acc, best_epoch_num, start_acc_test, start_acc_train, end_acc_test, end_acc_train = train_and_evaluate(Data_train[:, 0:2], Data_test[:,0:2], Data_train[:, 2], Data_test[:,2], hidden_neurons, exp=experiment, random_state=run)
        train_accuracies_all.append(train_accuracies)
        test_accuracies_all.append(test_accuracies)

        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(train_accuracies_all[run]) + 1), train_accuracies_all[run], label=f"Run {run+1} Train")
        plt.plot(range(1, len(test_accuracies_all[run]) + 1), test_accuracies_all[run], label=f"Run {run+1} Test")
        plt.title(f'Acc. Changes Over Epochs - Exp. {experiment+2} Data, Run={run+1}')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        #plt.show()
        
        #decision_boundary(exp=experiment, best_epoch_num=best_epoch_num, accuracy=best_epoch_acc, num_run=run, data_train=Data_train, start_acc_test =start_acc_test,  end_acc_test =end_acc_test, start_acc_train=start_acc_train, end_acc_train=end_acc_train)
        new_row_test = {'Run': run+1, 'AccStart': start_acc_test, 'AccBest': best_epoch_acc, 'AccBestEpochNr': best_epoch_num, 'AccEnd': end_acc_test}
        df_test = df_test._append(new_row_test, ignore_index=True)
        print(df_test)
        
        new_row_train = {'Run': run+1, 'AccStart': start_acc_train, 'AccBest': best_epoch_acc, 'AccBestEpochNr': best_epoch_num, 'AccEnd': end_acc_train}
        df_train = df_test._append(new_row_train, ignore_index=True)
        
    df_test.to_csv(f"test_exp_4_table_exp_{experiment+2}_data.csv",   index=False)
    df_train.to_csv(f"train_exp_4_table_exp_{experiment+2}_data.csv", index=False)


def decision_boundary(exp, best_epoch_num, accuracy, num_run, data_train, start_acc_test=None, start_acc_train=None, end_acc_train=None, end_acc_test=None)->None:
    fig, ax = plt.subplots(3, 2, figsize=(10, 20)) #Train 0, Test 1
    model_start_epoch = model_load(f'mlp_moodel_exp_{exp+2}_EPOCH_0.joblib')
    model_last_epoch  = model_load(f'mlp_moodel_exp_{exp+2}_EPOCH_LAST.joblib')
    model_best_epoch  = model_load(f'mlp_model_exp_{exp+2}_BEST.joblib')
    
    plot_decision_boundary_ax(Data_test[:,0:2], axes_dec=ax[0, 0], func=lambda X: model_best_epoch.predict(X))
    ax[0, 0].set_title(f"(Run={num_run}, Test exp={exp+2}) Best (Acc={accuracy}, Epoch={best_epoch_num})")
    plot_decision_boundary_ax(Data_test[:,0:2], axes_dec=ax[1, 0], func=lambda X: model_start_epoch.predict(X))
    ax[1, 0].set_title(f"(Run={num_run}, Test exp={exp+2}) Start (Acc={start_acc_test})")
    plot_decision_boundary_ax(Data_test[:,0:2], axes_dec=ax[2, 0], func=lambda X: model_last_epoch.predict(X))
    ax[2, 0].set_title(f"(Run={num_run}, Test exp={exp+2}) Last (Acc={end_acc_test})")

    plot_decision_boundary_ax(data_train[:,0:2], axes_dec=ax[0, 1], func=lambda X: model_best_epoch.predict(X))
    ax[0, 1].set_title(f"(Run={num_run}, Train exp={exp+2}) Best (Acc={accuracy}, Epoch={best_epoch_num})")
    plot_decision_boundary_ax(data_train[:,0:2], axes_dec=ax[1, 1], func=lambda X: model_start_epoch.predict(X))
    ax[1, 1].set_title(f"(Run={num_run}, Train exp={exp+2}) Start (Acc={start_acc_train})")
    plot_decision_boundary_ax(data_train[:,0:2], axes_dec=ax[2, 1], func=lambda X: model_last_epoch.predict(X))
    ax[2, 1].set_title(f"(Run={num_run}, Train exp={exp+2}) Last (Acc={end_acc_train})")
    plt.show()

if __name__ == "__main__":
    
    Data = np.genfromtxt(f"C:\\Users\\Michał\\Documents\\STUDIA\\II stopień, Informatyka Stosowana - inżynieria oprogramowania i uczenie maszynowe\\I sem\\Obliczenia inteligentne\\Projekt 1 - zadanie 2\\2_3.csv", delimiter=';')
    Data_test  = Data[260:300,:]
    Data_train_exp2 = Data[0:260,:]
    Data_train_exp3 = Data[0:60,:]

#Przypadek z eksperymentu 2
    run_random_state(num_runs=10, Data_train=Data_train_exp2, experiment=0)

#Przypadek z eksperymentu 3
    run_random_state(num_runs=10, Data_train=Data_train_exp3, experiment=1)

    
    
       

    