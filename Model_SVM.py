from sklearn.svm import SVC  # "Support Vector Classifier"
import numpy as np

from Evaluation import evaluation


def Model_SVM(train_data, train_target, test_data, test_target):
    print('SVM')
    IMG_SIZE = 10
    Train_Temp = np.zeros((train_data.shape[0], IMG_SIZE))
    for i in range(train_data.shape[0]):
        Train_Temp[i, :] = np.resize(train_data[i], IMG_SIZE)
    train_data = Train_Temp.reshape(Train_Temp.shape[0], IMG_SIZE)

    Test_Temp = np.zeros((test_data.shape[0], IMG_SIZE))
    for i in range(test_data.shape[0]):
        Test_Temp[i, :] = np.resize(test_data[i], IMG_SIZE)
    test_data = Test_Temp.reshape(Test_Temp.shape[0], IMG_SIZE)

    clf = SVC(kernel='linear')

    pred = np.zeros(test_target.shape)

    # fitting x samples and y classes
    for i in range(test_target.shape[1]):
        print(i, test_target.shape[1])
        clf.fit(train_data.tolist(), train_target[:, i].tolist())

        Y_pred = clf.predict(test_data.tolist())
        pred[:, i] = np.asarray(Y_pred)

    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0
    Eval = evaluation(test_target, pred.reshape(-1, 1))
    return Eval
