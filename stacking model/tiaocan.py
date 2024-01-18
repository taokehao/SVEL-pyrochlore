import numpy as np
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import KFold
import catboost as cat
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier


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
    return data_mat, lable_mat



if __name__ == '__main__':
    x_data, y_data = load_data("../feature_train2.txt", "../label_train.txt")
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
    x_data = np.asarray(x_data)
    y_data = np.asarray(y_data)
    smo = SMOTE(sampling_strategy='auto',
                random_state=0,  ## 随机器设定
                k_neighbors=10,  ## 用相近的 5 个样本（中的一个）生成正样本
                n_jobs=1,  ## 使用的例程数，为-1时使用全部CPU
                )
    x_data, y_data = smo.fit_resample(x_data, y_data)


    # print("请输入KFlod的值:", end='')
    # num = int(input())

    print(x_data.shape, y_data.shape)

    # 分类器使用 xgboost
    # model = XGBClassifier(objective="binary:logistic")
    # model = AdaBoostClassifier(random_state=0)
    model = DecisionTreeClassifier(random_state=0)

    # 设定搜索的xgboost参数搜索范围，值搜索XGBoost的主要6个参数
    param_dist = {
        # 'eta': np.linspace(0.01, 0.5, 500),
        # 'min_child_weight': range(1, 5, 1),
        # 'max_depth':range(3, 10, 1),

        # "n_estimators": range(30, 100, 1),
        # "learning_rate": np.linspace(0.01, 1, 100)

        'max_depth': range(1, 100, 1),
        'min_samples_leaf': range(1, 100, 1),
        'min_impurity_decrease': np.linspace(0.01, 2, 100)
    }

    # RandomizedSearchCV参数说明，clf1设置训练的学习器
    # param_dist字典类型，放入参数搜索范围
    # scoring = 'neg_log_loss'，精度评价方式设定为“neg_log_loss“
    # n_iter=300，训练300次，数值越大，获得的参数精度越大，但是搜索时间越长
    # n_jobs = -1，使用所有的CPU进行训练，默认为1，使用1个CPU
    cv = KFold(n_splits=10, shuffle=True, random_state=2)
    randn = RandomizedSearchCV(model, param_dist, scoring='accuracy', cv=cv, verbose=0, n_jobs=-1)

    # 在训练集上训练
    randn.fit(x_data, np.ravel(y_data))
    # 返回最优的参数
    print("best_params_:")
    print(randn.best_params_)
    # 输出最优训练器的精度
    print("best_score_:")
    print(randn.best_score_)
    # best_estimator_ = grid.best_estimator_
    # print("best_estimator_:")
    # print(best_estimator_)
    # joblib.dump(best_estimator_, '../model/catboost_model.pkl')

    data = open("./Optimal Parameter/" + model.__class__.__name__ + ".txt", "a")
    print("\nbest_params_:", file=data)
    print(randn.best_params_, file=data)
    print("best_score_:", file=data)
    print(randn.best_score_, file=data)
    data.close()

