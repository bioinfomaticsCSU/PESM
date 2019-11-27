import numpy as np
import os
from datetime import datetime
from sklearn import preprocessing
from sklearn.externals import joblib
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn import naive_bayes
from sklearn import tree
from sklearn import metrics
from sklearn.model_selection import StratifiedShuffleSplit
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from scipy import interp





def cal_index_prob(predict_prob, labels):
    precision, recall, pr_thresholds = metrics.precision_recall_curve(labels, predict_prob)
    all_F_measure = np.zeros(len(pr_thresholds))
    for k in range(0, len(pr_thresholds)):
        if (precision[k] + precision[k]) > 0:
            all_F_measure[k] = 2 * precision[k] * recall[k] / (precision[k] + recall[k])
        else:
            all_F_measure[k] = 0
    max_index = all_F_measure.argmax()
    threshold = pr_thresholds[max_index]

    predicted_score = np.zeros(len(labels))
    predicted_score[predict_prob > threshold] = 1

    f1 = metrics.f1_score(labels, predicted_score)
    accuracy = metrics.accuracy_score(labels, predicted_score)

    return accuracy, f1



def k_fold_eval():
    data = np.load("dataset.npz")
    ratio = 0.8
    times = 200


    x = data['X']
    y = data['Y']

    type_m = {"PESM": XGBClassifier(max_depth=6, learning_rate=0.01, n_estimators=1000)}

    for t in type_m:
        for i in range(1,times):
            sss = StratifiedShuffleSplit(n_splits=5, test_size=1-ratio,random_state=int(i))
            print("method:"+t)

            all_acc, all_pre, all_rec, all_f1, all_auc = list(), list(), list(), list(), list()
            for train_index, test_index in sss.split(x, y):
                x_train, x_test = x[train_index], x[test_index]
                y_train, y_test = y[train_index], y[test_index]
                start = datetime.now()
                model = type_m[t]
                model.fit(x_train, y_train)

                predictpro = model.predict_proba(x_test)
                acc, f1 = cal_index_prob(predictpro[:, 1], np.array(y_test))
                all_acc.append(acc)
                all_f1.append(f1)
                fpr, tpr, thr = metrics.roc_curve(np.array(y_test),predictpro[:, 1] )
                auc = metrics.auc(fpr, tpr)
                all_auc.append(auc)

                print("Testing example:%d, Accuracy:%g, f1:%g, auc:%g"
                      % (len(x_test), acc, f1, auc))
                end = datetime.now()
                print((end-start))



        print("Average accuracy:{:.6f}, f1:{:.6f}, auc:{:.6f}\n".
            format(np.average(all_acc), np.average(all_f1), np.average(all_auc)))

k_fold_eval()
