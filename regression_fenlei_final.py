import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier as GBDT
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import KFold


def load_data(x_path, y_path, z_path):
    num_feat = len(open(x_path).readline().split(" "))
    data_mat = list()
    lable_mat = list()
    pred_mat = list()
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
        cur_line = line.strip().split("\t")
        lable_mat.append(float(cur_line[0]))
    # 读取标签
    z_fr = open(z_path)
    for line in z_fr.readlines():
        line_arr = list()
        cur_line = line.strip().split("  ")
        for i in range(num_feat):
            line_arr.append(float(cur_line[i]))
        pred_mat.append(line_arr)
    return data_mat, lable_mat, pred_mat


def print_sample(pred_0list):
    readlines = open("data.txt").readlines()
    for i in pred_0list:
        data = open("data_screen.txt", "a")
        print(readlines[i], file=data, end="")
        data.close()


if __name__ == '__main__':
    x_data, y_data, x_pred = load_data("feature_train2.txt", "label_train.txt", "feature_test2.txt")
    # # 线性回归
    # index = 0
    # num_0 = 0
    # for i in y_data:
    #     if i < 0.5:
    #         y_data[index] = 0
    #         num_0 += 1
    #     else:
    #         y_data[index] = 1
    #     index += 1
    # # print(y_data)
    # print(len(y_data), "训练集中有", num_0, "个0标签", 144 - num_0, "个1标签\n")
    #
    # # GBDT
    # # print("请输入KFlod的值:", end='')
    # # num = int(input())
    #
    # print('GBDT结果为：')
    # clf = GBDT(loss='log_loss',  # 损失函数默认deviance  deviance具有概率输出的分类的偏差
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
    # plot_regression(clf, x_data, y_data, x_pred)

    model = joblib.load('./stacking model/model/vote.pkl')
    model.predict(x_pred)
    y_pred = model.predict(x_pred)

    # 展示预测结果
    pred_0list = []
    print("\n训练预测成功，下面进行统计\n")
    num_0 = 0
    num_1 = 0
    cout = 0  # 用于计数
    for i in y_pred:
        if i == 0:
            num_0 += 1
            pred_0list.append(cout)
        else:
            num_1 += 1
        cout += 1
    print("带隙0个数：", num_0, " 带隙1个数：", num_1)
    print("\n0标签的序号：\n", pred_0list)

    # 筛选出带隙0的输出到data_screen.txt
    print_sample(pred_0list)

    # 然后进行特征重要性筛选排序
    feature_importances = []
    feature_importances_not0 = []
    for i in model.feature_importances_:
        feature_importances.append(i)
        if i != 0:
            feature_importances_not0.append(i)
    feature_importances_not0.sort(key=None, reverse=True)

    importance_dick = {}
    for i in feature_importances_not0:
        index = 0
        for j in feature_importances:
            if i == j:
                importance_dick[index] = i
            index += 1
    print("\n特征重要性排序结果 (120列特征的序号，从0开始):")

    for i in importance_dick:
        print(i, ": ", importance_dick.get(i))

    print("-------------")
