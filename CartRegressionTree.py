# !user/bin/python
# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import treePlotter
import matplotlib as mpl
import matplotlib.pyplot as plt


# 最优切分点s
def value_split(x, y):
    m = len(x)
    error_min = np.inf
    for i in range(m):
        # 计算左切分区域误差和
        c1_sum = sum([y[j] for j in range(i + 1)])
        c1 = float(c1_sum)/(i + 1)
        c1_error = sum([pow(y[j] - c1, 2) for j in range(i + 1)])
        # 计算右切分区域误差和
        if (m - (i + 1)) == 0:
            c2_error = 0
        else:
            c2_sum = sum([y[-z-1] for z in range(m - (i + 1))])
            c2 = float(c2_sum)/(m - (i + 1))
            c2_error = sum([pow(y[-z-1] - c2, 2) for z in range(m - (i + 1))])
        # 记录最小误差和
        error_sum = c1_error + c2_error
        if error_sum < error_min:
            error_min = error_sum
            split_point = x[i]
    return error_min, split_point


# 最优切分特征
def split_feature(x):
    n = len(x[0])
    min_feature_error = np.inf
    for i in range(n-1):
        index = np.argsort(x[:, i])
        x_sort = x[:, i][index]
        y_sort = x[:, -1][index]
        error_feat, split_point = value_split(x_sort, y_sort)

        if error_feat < min_feature_error:
            min_feature_error = error_feat
            min_feature_index = i
            min_split_point = split_point
    return min_feature_error, min_feature_index, min_split_point


# 分割区域
def split_data(x_fa, feature, split_point):
    list_feature = [x[feature] for x in x_fa]
    list_feature.reverse()
    m = list_feature.index(split_point) + 1
    index = np.argsort(x_fa[:, feature])

    # 分割
    x_delete = np.delete(x_fa, feature, axis=1)[index]
    x_r1 = x_delete[:m]
    x_r2 = x_delete[m:]

    # 计算输出值
    y_r1 = x_r1[:, -1]
    c_r1 = float(sum(y_r1))/len(x_r1)
    y_r2 = x_r2[:, -1]
    if len(x_r2) == 0:
        c_r2 = 0
    else:
        c_r2 = float(sum(y_r2)) / len(x_r2)
    return x_r1, c_r1, x_r2, c_r2


# 生成树
def create_tree(x, c_r):
    if len(x[0]) == 1:
        return float(sum(x[:, -1]))/len(x[:, -1])
    if len(x) == 1:
        return x[:, -1][0]
    min_feat_error, min_feat_index, min_feat_point = split_feature(x)
    best_key = (min_feat_index, min_feat_point, c_r)
    my_tree = {best_key: {}}
    x_r1, c_r1, x_r2, c_r2 = split_data(x, min_feat_index, min_feat_point)
    if len(x_r1) == 0:
        return c_r1
    else:
        my_tree[best_key][0] = create_tree(x_r1, c_r1)
    if len(x_r2) == 0:
        return c_r2
    else:
        my_tree[best_key][1] = create_tree(x_r2, c_r2)

    return my_tree


# 预测结果
def predict(tree, x):
    first_str = tree.keys()[0]
    second_dict = tree[first_str]
    feature_index = first_str[0]
    if x[feature_index] <= first_str[1]:
        if type(second_dict[0]).__name__ == 'dict':
            class_label = predict(second_dict[0], x)
        else:
            class_label = second_dict[0]
    if x[feature_index] > first_str[1]:
        if type(second_dict[1]).__name__ == 'dict':
            class_label = predict(second_dict[1], x)
        else:
            class_label = second_dict[1]

    return class_label

if __name__ == "__main__":
    data = pd.read_table('housing.data', header=None, sep='\s+')
    x_data, y_data = np.array(data.ix[:, :12]), np.array(data.ix[:, 13])
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size=0.7, random_state=0)

    y_train = y_train.reshape((len(x_train), 1))
    c_ave = float(sum(y_train))/len(y_train)
    data_train = np.hstack((x_train, y_train))
    # 构建树
    my_tree = create_tree(data_train, c_ave)
    # 画树
    treePlotter.createPlot(my_tree)

    # 预测结果
    y_pre = []
    for i in range(len(x_test)):
        test_vec = x_test[i]
        result = predict(my_tree, test_vec)
        y_pre.append(result)

    # 对比图
    mpl.rcParams['font.sans-serif'] = [u'SimHei']  # 黑体 FangSong/KaiTi
    mpl.rcParams['axes.unicode_minus'] = False
    t = range(len(x_test))
    plt.figure(figsize=(10, 8), facecolor='w')
    plt.plot(t, y_pre, color='r', lw=1.5, label=u'预测值')
    plt.plot(t, y_test, color='g', lw=1.5, label=u'真实值')
    plt.xlabel('X', fontsize=2.8)
    plt.ylabel('Y', fontsize=2.8)
    plt.legend(loc='best')
    plt.show()














