# 对现在的结果进行分析
readlines1 = open("fenlei_1.txt").readlines()
readlines2 = open("feature-pre.txt").readlines()

for i in readlines1:
    if i not in readlines2:
        data = open("fenlei_1_2.txt", "a")
        print(i, file=data, end="")
        data.close()


# import sklearn
#
# print(sklearn.__version__)

# 过采样
# from imblearn.over_sampling import SMOTE
# import numpy as np
# x_data = np.loadtxt("feature_train2.txt")
# y_data = np.loadtxt("label_train.txt")
# index = 0
# num_0 = 0
# for i in y_data:
#     if i < 0.5:
#         y_data[index] = 0
#         num_0 += 1
#     else:
#         y_data[index] = 1
#     index += 1
# print(len(y_data), num_0, 144 - num_0)
# print(y_data)
#
# smo = SMOTE(random_state=10)
# x_smo, y_smo = smo.fit_resample(x_data, y_data)
# print(len(x_smo))
# print(len(y_smo))
#
# count = 0
# for i in y_smo:
#     if i == 1:
#         count += 1
# print(count)
