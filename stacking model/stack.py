# 常用工具库
import re
import numpy as np
import pandas as pd
import matplotlib as mlp
import matplotlib.pyplot as plt
import time

# 算法辅助 & 数据
import sklearn
from sklearn.model_selection import KFold, cross_validate
from sklearn.datasets import load_digits  # 分类 - 手写数字数据集
from sklearn.datasets import load_iris

from sklearn.model_selection import train_test_split

# 算法（单一学习器）
from sklearn.neighbors import KNeighborsClassifier as KNNC
from sklearn.neighbors import KNeighborsRegressor as KNNR
from sklearn.tree import DecisionTreeRegressor as DTR
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.linear_model import LinearRegression as LR
from sklearn.linear_model import LogisticRegression as LogiR
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.ensemble import GradientBoostingRegressor as GBR
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb

# 融合模型
from sklearn.ensemble import StackingClassifier

data = load_digits()
X = data.data
y = data.target

# 划分数据集
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.2, random_state=1412)


def fusion_estimators(clf):
    """
    对融合模型做交叉验证，对融合模型的表现进行评估
    """
    cv = KFold(n_splits=5, shuffle=True, random_state=1412)
    results = cross_validate(clf, Xtrain, Ytrain
                             , cv=cv
                             , scoring="accuracy"
                             , n_jobs=-1
                             , return_train_score=True
                             , verbose=False)
    test = clf.fit(Xtrain, Ytrain).score(Xtest, Ytest)
    print("train_score:{}".format(results["train_score"].mean())
          , "\n cv_mean:{}".format(results["test_score"].mean())
          , "\n test_score:{}".format(test)
          )


def individual_estimators(estimators):
    """
    对模型融合中每个评估器做交叉验证，对单一评估器的表现进行评估
    """
    for estimator in estimators:
        cv = KFold(n_splits=5, shuffle=True, random_state=1412)
        results = cross_validate(estimator[1], Xtrain, Ytrain
                                 , cv=cv
                                 , scoring="accuracy"
                                 , n_jobs=-1
                                 , return_train_score=True
                                 , verbose=False)
        test = estimator[1].fit(Xtrain, Ytrain).score(Xtest, Ytest)
        print(estimator[0]
              , "\n train_score:{}".format(results["train_score"].mean())
              , "\n cv_mean:{}".format(results["test_score"].mean())
              , "\n test_score:{}".format(test)
              , "\n")


# 逻辑回归没有增加多样性的选项
clf1 = LogiR(max_iter=3000, C=0.1, random_state=1412, n_jobs=8)
# 增加特征多样性与样本多样性
clf2 = RFC(n_estimators=100, max_features="sqrt", max_samples=0.9, random_state=1412, n_jobs=8)
# 特征多样性，稍微上调特征数量
clf3 = GBC(n_estimators=100, max_features=16, random_state=1412)

# 增加算法多样性，新增决策树与KNN
clf4 = DTC(max_depth=8, random_state=1412)
clf5 = KNNC(n_neighbors=10, n_jobs=8)
clf6 = GaussianNB()

# 新增随机多样性，相同的算法更换随机数种子
clf7 = RFC(n_estimators=100, max_features="sqrt", max_samples=0.9, random_state=4869, n_jobs=8)
clf8 = GBC(n_estimators=100, max_features=16, random_state=4869)

estimators = [("Logistic Regression", clf1), ("RandomForest", clf2)
    , ("GBDT", clf3), ("Decision Tree", clf4), ("KNN", clf5)
              # , ("Bayes",clf6)
    , ("RandomForest2", clf7), ("GBDT2", clf8)
              ]

# 选择单个评估器中分数最高的随机森林作为元学习器
# 也可以尝试其他更简单的学习器
final_estimator = RFC(n_estimators=100
                      , min_impurity_decrease=0.0025
                      , random_state= 420, n_jobs=8)
clf = StackingClassifier(estimators=estimators #level0的7个体学习器
                         ,final_estimator=final_estimator #level 1的元学习器
                         ,n_jobs=8)

fusion_estimators(clf)
fusion_estimators(clf)
