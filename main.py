import numpy as np
import matplotlib.pylab as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold

def std(arr):
    z   = np.mean(arr)
    std = np.std(arr)
    return [(x - z) / std for x in arr]

def two_classify():
    file = open('PH2Dataset/PH2.csv')
    data = file.readlines()
    y_data = np.zeros(len(data))
    for index, line in enumerate(data):
        if (line.split(',')[3] == 'X\n'):
            y_data[index] = 0
        else:
            y_data[index] = 1
        '''
        if (line.split(',')[1] == 'X'):
            y_data[index] = 1
        if (line.split(',')[2] == 'X'):
            y_data[index] = 1
        '''
    x_data = np.load('feature.npy')

    for i in range(x_data.shape[1]):
        x_data[:, i] = std(x_data[:, i])

    kf = StratifiedKFold(n_splits = 5, shuffle = True)
    idx = 1
    for train_index, test_index in kf.split(x_data, y_data):
        print("===== Round %d =====" % idx)
        idx = idx + 1
        x_train, x_valid = x_data[train_index], x_data[test_index]
        y_train, y_valid = y_data[train_index], y_data[test_index]
        weights = np.zeros(len(y_train))
        weights[y_train == 0] = 4
        weights[y_train == 1] = 1

        c1 = LogisticRegression(class_weight = {0:4, 1:1})
        c2 = RandomForestClassifier(class_weight = {0:4, 1:1})
        c3 = GradientBoostingClassifier()
        c4 = GaussianNB(priors = [0.8, 0.2])

        c1.fit(x_train, y_train)
        c2.fit(x_train, y_train)
        c3.fit(x_train, y_train, sample_weight = weights)
        c4.fit(x_train, y_train)
        y_pred_1 = c1.predict_proba(x_valid)[:, 1]
        y_pred_2 = c2.predict_proba(x_valid)[:, 1]
        y_pred_3 = c3.predict_proba(x_valid)[:, 1]
        y_pred_4 = c4.predict_proba(x_valid)[:, 1]
        fpr_1, tpr_1, _ = roc_curve(y_valid, y_pred_1)
        fpr_2, tpr_2, _ = roc_curve(y_valid, y_pred_2)
        fpr_3, tpr_3, _ = roc_curve(y_valid, y_pred_3)
        fpr_4, tpr_4, _ = roc_curve(y_valid, y_pred_4)

        print('The result of logisitic reg : %.4f, auc: %.4f' % (c1.score(x_valid, y_valid), auc(fpr_1, tpr_1)))
        print('The result of reandom forest: %.4f, auc: %.4f' % (c2.score(x_valid, y_valid), auc(fpr_2, tpr_2)))
        print('The result of gradient boost: %.4f, auc: %.4f' % (c3.score(x_valid, y_valid), auc(fpr_3, tpr_3)))
        print('The result of naive bayes:    %.4f, auc: %.4f' % (c4.score(x_valid, y_valid), auc(fpr_4, tpr_4)))

    print(c3.feature_importances_)
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_1, tpr_1, label='LR')
    plt.plot(fpr_2, tpr_2, label='RF')
    plt.plot(fpr_3, tpr_3, label='GB')
    plt.plot(fpr_4, tpr_4, label='NB')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.show()

def three_classify():
    file = open('PH2Dataset/PH2.csv')
    data = file.readlines()
    y_data = np.zeros(len(data))
    for index, line in enumerate(data):
        if (line.split(',')[1] == 'X'):
            y_data[index] = 0
        if (line.split(',')[2] == 'X'):
            y_data[index] = 1
        if (line.split(',')[3] == 'X\n'):
            y_data[index] = 2
    x_data = np.load('feature.npy')

    for i in range(x_data.shape[1]):
        x_data[:, i] = std(x_data[:, i])

    kf = StratifiedKFold(n_splits = 5, shuffle = True)
    idx = 1
    for train_index, test_index in kf.split(x_data, y_data):
        print("===== Round %d =====" % idx)
        idx = idx + 1
        x_train, x_valid = x_data[train_index], x_data[test_index]
        y_train, y_valid = y_data[train_index], y_data[test_index]
        weights = np.zeros(len(y_train))
        weights[y_train == 0] = 1
        weights[y_train == 1] = 1
        weights[y_train == 2] = 2

        c1 = LogisticRegression(class_weight = {0:1, 1:1, 2:2})
        c2 = RandomForestClassifier(class_weight = {0:1, 1:1, 2:2})
        c3 = GradientBoostingClassifier()
        c4 = GaussianNB(priors = [0.4, 0.4, 0.2])

        c1.fit(x_train, y_train)
        c2.fit(x_train, y_train)
        c3.fit(x_train, y_train, sample_weight = weights)
        c4.fit(x_train, y_train)
        print('The result of logisitic reg : %.4f' % (c1.score(x_valid, y_valid)))
        print('The result of reandom forest: %.4f' % (c2.score(x_valid, y_valid)))
        print('The result of gradient boost: %.4f' % (c3.score(x_valid, y_valid)))
        print('The result of naive bayes:    %.4f' % (c4.score(x_valid, y_valid)))

    print(c3.feature_importances_)

two_classify()
three_classify()

