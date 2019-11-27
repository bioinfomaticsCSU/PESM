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
import matplotlib.pyplot as pyplot
from scipy import interp
from xgboost import plot_importance

data = np.load("dataset.npz")

x = data['X']
y = data['Y']

model = XGBClassifier()
model.fit(x, y)

# feature importance
print(model.feature_importances_)


# plot
firms = ['%CC in mat', '%CG in mat', '%CU in mat',
    '%GC in mat','%GG in mat','%GU in mat',
    '%UC in mat','%UG in mat','%UU in mat',
    '%CC in pre','%CG in pre','%CU in pre',
    '%GC in pre','%GG in pre','%GU in pre',
    '%UC in pre','%UG in pre','%UU in pre',
    'P(s)      ','nP(s)', 'Q(s)','nQ(s)','D(s)','nD(s)',
    'U in pre','C in pre','G in pre',
    'MIR length','U in MIR','C in MIR','G in MIR',
    'non-MIR length','U in non-MIR','C in non-MIR','G in non-MIR',
    'MFE    ','nMFE','cleavage site']
pyplot.bar(range(len(model.feature_importances_)), model.feature_importances_)
x_axs=[i for i in range(len(model.feature_importances_))]
pyplot.xticks(x_axs,firms,rotation=45)
pyplot.show()
