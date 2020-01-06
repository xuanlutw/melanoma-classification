import numpy as np
import matplotlib.pylab as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold
from joblib import dump, load
from img_process import *
import sys
import os

def cal_std_parameter(arr):
    res = [[np.mean(x), np.std(x)] for x in np.swapaxes(arr, 0, 1)]
    np.save('std_para', res)
    return res

def cal_std(arr, std_para):
    return np.asarray([list(map(lambda x: (x[0] - x[1][0]) / x[1][1], zip(x, std_para))) for x in arr])

def train_two_classify():
    file = open('PH2Dataset/PH2.csv')
    data = file.readlines()
    y_data = np.zeros(len(data))
    for index, line in enumerate(data):
        if (line.split(',')[3] == 'X\n'):
            y_data[index] = 0
        else:
            y_data[index] = 1
    x_data = np.load('feature.npy')

    std_para = cal_std_parameter(x_data)
    x_data   = cal_std(x_data, std_para)

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

    dump(c1, 'two1.joblib')
    dump(c2, 'two2.joblib')
    dump(c3, 'two3.joblib')
    dump(c4, 'two4.joblib')
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

def train_three_classify():
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

    std_para = cal_std_parameter(x_data)
    x_data   = cal_std(x_data, std_para)

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
    dump(c1, 'three1.joblib')
    dump(c2, 'three2.joblib')
    dump(c3, 'three3.joblib')
    dump(c4, 'three4.joblib')
    print(c3.feature_importances_)

def valid(img_path):
    std_para = np.load('std_para.npy')
    segmentation(img_path, 'valid')
    feat = [feature_extraction(img_path, "data/valid-bw.bmp")]
    std_para = np.load('std_para.npy')
    feat = cal_std(feat, std_para)
    c1 = load('two1.joblib')
    c2 = load('two2.joblib')
    c3 = load('two3.joblib')
    c4 = load('two4.joblib')
    to_name = (lambda x: "melanoma" if x == 0 else "not melanoma")
    print("===== Two =====")
    print('The result of logisitic reg : %s' % to_name(c1.predict(feat)))
    print('The result of reandom forest: %s' % to_name(c2.predict(feat)))
    print('The result of gradient boost: %s' % to_name(c3.predict(feat)))
    print('The result of naive bayes:    %s' % to_name(c4.predict(feat)))

    c1 = load('three1.joblib')
    c2 = load('three2.joblib')
    c3 = load('three3.joblib')
    c4 = load('three4.joblib')
    to_name = (lambda x: "common nevus" if x == 0 else ("atypical nevus" if x == 1 else "melanoma"))
    print("===== Three =====")
    print('The result of logisitic reg : %s' % to_name(c1.predict(feat)))
    print('The result of reandom forest: %s' % to_name(c2.predict(feat)))
    print('The result of gradient boost: %s' % to_name(c3.predict(feat)))
    print('The result of naive bayes:    %s' % to_name(c4.predict(feat)))


#train_two_classify()
#train_three_classify()
#valid('PH2Dataset/PH2 Dataset images/IMD024/IMD024_Dermoscopic_Image/IMD024.bmp')
try:
    os.mkdir('data')
except:
    pass
valid(sys.argv[1])
