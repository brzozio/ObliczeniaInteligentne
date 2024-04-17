"""
    -> generates csv of accuracy scores where rows are continuing values and columns are a flattened tree
                            2_2.csv                  2_3.csv
                       knn    svc    mlp        knn    svc    mlp
                   train test             ...

    -> generates 2 CSV files of shape:
                     X_1     X_2     Y_true   Y_knn_(min/best/max)   Y_svc_(min/best/max)   Y_mlp_(min/best/max)
        train   0
                ...
        test    240
                ...
                299

"""
import numpy as np
from sklearn import metrics as skmet
from sklearn import model_selection as skms
from sklearn import neural_network as sknn
from sklearn import neighbors as sknb
from sklearn import svm as sksvm

TrainData: np.array = np.zeros((2, 240, 3))
TestData: np.array = np.zeros((2, 60, 3))
ReadData = np.genfromtxt("2_2.csv", delimiter=";")
TrainData[0, :, 0:2], TestData[0, :, 0:2], TrainData[0, :, 2], TestData[0, :, 2] = skms.train_test_split(
    ReadData[:, 0:2], ReadData[:, 2], train_size=240, random_state=42)
ReadData = np.genfromtxt("2_3.csv", delimiter=";")
TrainData[1, :, 0:2], TestData[1, :, 0:2], TrainData[1, :, 2], TestData[1, :, 2] = skms.train_test_split(
    ReadData[:, 0:2], ReadData[:, 2], train_size=240, random_state=42)
# print(TrainData)

accScore: np.array = np.zeros((2, 3, 2, 10))  # [csv, model, dataset, variable]
modelOut: np.array = np.zeros((2, 300, 3 + 3 * 3))
modelOut[:, 0:240, 0:3] = TrainData
modelOut[:, 240:300, 0:3] = TestData

params = [2, 3, 4, 5, 6, 8, 10, 20, 30, 50]

for i in range(2):
    for k in range(3):
        tempMaxAcc = 0.0
        for j in range(10):
            write_flag = False
            match k:
                case 0:
                    model = sknb.KNeighborsClassifier(n_neighbors=params[j])
                case 1:
                    model = sksvm.SVC(kernel='rbf', C=(j + 1.0) / 10.0)
                case 2:
                    model = sknn.MLPClassifier(hidden_layer_sizes=params[j], activation='relu',
                                               max_iter=100000, tol=0, n_iter_no_change=100000, solver='sgd')

            model.fit(TrainData[i, :, 0:2], TrainData[i, :, 2])

            ModelOutTemp = model.predict(TestData[i, :, 0:2])
            if j == 0:
                modelOut[i, 240:300, 3 + 3 * k] = ModelOutTemp
            if j == 9:
                modelOut[i, 240:300, 3 + 3 * k + 2] = ModelOutTemp
            accScore[i, k, 1, j] = skmet.accuracy_score(TestData[i, :, 2], ModelOutTemp)
            if tempMaxAcc < accScore[i, k, 1, j]:
                tempMaxAcc = accScore[i, k, 1, j]
                write_flag = True
                modelOut[i, 240:300, 3 + 3 * k + 1] = ModelOutTemp

            ModelOutTemp = model.predict(TrainData[i, :, 0:2])
            if j == 0:
                modelOut[i, 0:240, 3 + 3 * k] = ModelOutTemp
            if j == 9:
                modelOut[i, 0:240, 3 + 3 * k + 2] = ModelOutTemp
            accScore[i, k, 0, j] = skmet.accuracy_score(TrainData[i, :, 2], ModelOutTemp)
            if write_flag:
                modelOut[i, 0:240, 3 + 3 * k + 1] = ModelOutTemp

            print(str(i) + ", " + str(k) + ", " + str(j) + "done")

np.savetxt("exp2Outs_2_2.csv", modelOut[0], delimiter=";")
np.savetxt("exp2Outs_2_3.csv", modelOut[1], delimiter=";")
printableAcc = np.zeros((10, 2 * 3 * 2))
for i in range(2):
    for j in range(3):
        for k in range(2):
            printableAcc[:, i * 2 * 3 + j * 2 + k] = accScore[i, j, k, :]
np.savetxt("exp2Acc.csv", printableAcc, delimiter=";")
