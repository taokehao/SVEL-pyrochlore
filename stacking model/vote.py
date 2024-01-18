# 常用工具库
import csv
import re

import joblib
import numpy as np
import pandas as pd
import matplotlib as mlp
import matplotlib.pyplot as plt
import time

# 算法辅助 & 数据
import sklearn
from sklearn.model_selection import KFold, cross_validate
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
from sklearn.ensemble import GradientBoostingClassifier as GBDT
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier

# 融合模型
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import VotingRegressor
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report



def load_data(x_path, y_path):
    num_feat = len(open(x_path).readline().split(" "))
    data_mat = list()
    lable_mat = list()
    # 读取特征
    x_fr = open(x_path)
    for line in x_fr.readlines():
        line_arr = list()
        cur_line = line.strip().split(" ")
        for i in range(num_feat):
            line_arr.append(float(cur_line[i]))
        data_mat.append(line_arr)
    # 读取标签
    y_fr = open(y_path)
    for line in y_fr.readlines():
        line_arr = list()
        cur_line = line.strip().split("\t")
        lable_mat.append(float(cur_line[0]))
    return np.asarray(data_mat), np.asarray(lable_mat)


def individual_estimators(estimators, x_data, y_data):
    """
    对模型融合中每个评估器做交叉验证，对单一评估器的表现进行评估
    """
    for estimator in estimators:
        cv = KFold(n_splits=10, shuffle=True, random_state=2)
        results = cross_validate(estimator[1], x_data, y_data
                                 , cv=cv
                                 , scoring="accuracy"
                                 , n_jobs=-1
                                 , return_train_score=True
                                 , verbose=False)
        # test = estimator[1].fit(Xtrain,Ytrain).score(Xtest,Ytest)
        print(estimator[0]
              , "\n train_score:{}".format(results["train_score"].mean())
              , "\n cv_mean:{}".format(results["test_score"].mean())
              # ,"\n test_score:{}".format(test)
              , "\n")


def fusion_estimators(clf, x_data, y_data):
    """
    对融合模型做交叉验证，对融合模型的表现进行评估
    """
    cv = KFold(n_splits=10, shuffle=True, random_state=2)
    results = cross_validate(clf, x_data, y_data
                             , cv=cv
                             , scoring="accuracy"
                             , n_jobs=-1
                             , return_train_score=True
                             , verbose=False)
    # test = clf.fit(Xtrain,Ytrain).score(Xtest,Ytest)
    print("train_score:{}".format(results["train_score"].mean())
          , "\n cv_mean:{}".format(results["test_score"].mean())
          # ,"\n test_score:{}".format(test)
          )



if __name__ == '__main__':

    x_data, y_data = load_data("../feature_train2.txt", "../label_train.txt")
    print(x_data)
    print(y_data)
    # 线性回归
    index = 0
    num_0 = 0
    for i in y_data:
        if i < 0.5:
            y_data[index] = 0
            num_0 += 1
        else:
            y_data[index] = 1
        index += 1
    print(y_data)
    print(len(y_data), num_0, 147 - num_0)

    # 过采样
    smo = SMOTE(sampling_strategy='auto',
                random_state=0,  ## 随机器设定
                k_neighbors=10,  ## 用相近的 5 个样本（中的一个）生成正样本
                n_jobs=1,  ## 使用的例程数，为-1时使用全部CPU
                )
    x_data, y_data = smo.fit_resample(x_data, y_data)

    for i in x_data:
        newRow = i.tolist()
        csvFile = open("C:\\Users\\taozi\\Desktop\\x_data.csv", 'a', newline='', encoding='utf-8')
        writer = csv.writer(csvFile)
        writer.writerow(newRow)  # 数据写入文件中
        csvFile.close()
    # print("请输入KFlod的值:", end='')
    # num = int(input())

    # clf1 = LogiR(max_iter=3000, C=0.1, random_state=1412, n_jobs=8)  # 这一组合可能说明我们的max_iter设置得太大了
    # clf2 = RFC(n_estimators=100, max_depth=12, random_state=1412, n_jobs=8)
    # clf3 = GBDT(loss='log_loss',  # 损失函数默认deviance  deviance具有概率输出的分类的偏差
    #            n_estimators=42,  # 默认100 回归树个数 弱学习器个数
    #            learning_rate=0.313,  # 默认0.1学习速率/步长0.0-1.0的超参数  每个树学习前一个树的残差的步长0.1
    #            max_depth=3,  # 默认值为3每个回归树的深度  控制树的大小 也可用叶节点的数量max leaf nodes控制
    #            subsample=0.8750000000000001,  # 树生成时对样本采样 选择子样本<1.0导致方差的减少和偏差的增加
    #            min_samples_split=2,  # 生成子节点所需的最小样本数 如果是浮点数代表是百分比
    #            min_samples_leaf=1,  # 叶节点所需的最小样本数  如果是浮点数代表是百分比
    #            max_features=None,  # 在寻找最佳分割点要考虑的特征数量auto全选/sqrt开方/log2对数/None全选/int自定义几个/float百分比
    #            max_leaf_nodes=None,  # 叶节点的数量 None不限数量
    #            verbose=0,  # 打印输出 大于1打印每棵树的进度和性能
    #            warm_start=False,  # True在前面基础上增量训练(重设参数减少训练次数) False默认擦除重新训练
    #            random_state=0  # 随机种子-方便重现
    #            )
    # clf4 = KNNC()
    # clf5 = DTC(random_state=1412)
    # estimators = [
    #     # ("Logistic Regression", clf1),
    #     ("RandomForest", clf2),
    #     ("GBDT", clf3),
    #     # ('KNNC', clf4),
    #     ('DTC', clf5)
    # ]
    # clf = VotingClassifier(estimators, voting="soft")

    # 逻辑回归没有增加多样性的选项
    clf1 = LogiR(max_iter=3000, C=0.1, random_state=1412, n_jobs=8)
    # 增加特征多样性与样本多样性
    clf2 = RFC(n_estimators=100, max_features="sqrt", max_samples=0.9, random_state=0, n_jobs=8)
    # 特征多样性，稍微上调特征数量
    clf3 = GBDT(
        loss='log_loss',  # 损失函数默认deviance  deviance具有概率输出的分类的偏差
        n_estimators=42,  # 默认100 回归树个数 弱学习器个数
        learning_rate=0.313,  # 默认0.1学习速率/步长0.0-1.0的超参数  每个树学习前一个树的残差的步长0.1
        max_depth=3,  # 默认值为3每个回归树的深度  控制树的大小 也可用叶节点的数量max leaf nodes控制
        subsample=0.8750000000000001,  # 树生成时对样本采样 选择子样本<1.0导致方差的减少和偏差的增加
        min_samples_split=2,  # 生成子节点所需的最小样本数 如果是浮点数代表是百分比
        min_samples_leaf=1,  # 叶节点所需的最小样本数  如果是浮点数代表是百分比
        max_features=None,  # 在寻找最佳分割点要考虑的特征数量auto全选/sqrt开方/log2对数/None全选/int自定义几个/float百分比
        max_leaf_nodes=None,  # 叶节点的数量 None不限数量
        verbose=0,  # 打印输出 大于1打印每棵树的进度和性能
        warm_start=False,  # True在前面基础上增量训练(重设参数减少训练次数) False默认擦除重新训练
        random_state=0  # 随机种子-方便重现
    )

    # 增加算法多样性，新增决策树、KNN、贝叶斯
    clf4 = DTC(max_depth=8, random_state=0)
    clf5 = KNNC(n_neighbors=10, n_jobs=8)
    clf6 = GaussianNB()

    # 新增随机多样性，相同的算法更换随机数种子
    clf7 = RFC(n_estimators=100, max_features="sqrt", max_samples=0.9, random_state=0, n_jobs=8)
    clf8 = GBDT(random_state=0)
    clf9 = MultinomialNB(alpha=1.0
                         , fit_prior=True
                         , class_prior=None)
    clf10 = BernoulliNB(alpha=1.0
                        , binarize=0.0
                        , fit_prior=True
                        , class_prior=None)
    clf11 = XGBClassifier(objective="binary:logistic", eta=0.4793787575150301, max_depth=4, min_child_weight=1)
    clf12 = AdaBoostClassifier(random_state=0)

    estimators = [
        # ("Logistic Regression", clf1),
        # ("RandomForest", clf2),
        ("GBDT", clf3),  # 需要保留
        ("Decision Tree", clf4),  # 需要保留
        # ("KNN", clf5),
        ("Bayes", clf6),  # 需要保留
        # ("RandomForest2", clf7),
        # ("GBDT2", clf8),
        # ("MultinomialNB", clf9),
        ("BernoulliNB", clf10),  # 需要保留
        # ("Xgboost", clf11),
        ("AdaBoost", clf12)  # 需要保留
    ]
    clf = VotingClassifier(estimators, voting="soft")

    individual_estimators(estimators, x_data, y_data)
    fusion_estimators(clf, x_data, y_data)

    # clf = VotingClassifier(estimators, voting="soft")
    # clf.fit(x_data, y_data)
    # joblib.dump(clf, './model/vote.pkl')


