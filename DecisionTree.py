# !user/bin/python #
# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
import treePlotter


# 计算熵
def entropy(x, y):
    if x == 0:
        return 0
    else:
        value = float(x)/y
        return - value * np.log2(value)


# 计算经验熵
def emp_entropy(data, d_class, n_samples):
    ent_class = 0
    for x in range(d_class):
        class_sum = np.sum(data[:, -1-x])
        ent_class += entropy(class_sum, n_samples)
    return ent_class


# 计算条件经验熵
def entropy_condition(data, d_class, n_split):
    h_da = 0
    keys = set(data[:, n_split])
    for key in keys:
        new_list = np.array([x for x in data if x[n_split] == key])
        n_di = len(new_list)
        emp_di = emp_entropy(new_list, d_class, n_di)
        h_da += (float(n_di) / D) * emp_di
    return h_da


# 选择最好的特征分裂
def choose_best_feature(data, d_class):
    best_feature = -1
    best_info_gain = -1
    n_feature = len(data[0]) - d_class
    for i in range(n_feature):
        ent_feature = entropy_condition(data, d_class, i)
        info_gain = h_d - ent_feature
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = i
    return best_feature, best_info_gain


# 生成树
def create_tree(data, labels, d_class, name_class):
    tree_labels = labels[:]
    for x in range(d_class):
        if len(data[:, -1-x]) == np.sum(data[:, -1-x]):
            return name_class[-1-x]
    if len(data[0]) == d_class:
        maj = 0
        maj_class = ''
        for x in range(d_class):
            sum_class = np.sum(data[:, -1-x])
            if sum_class > maj:
                maj = sum_class
                maj_class = name_class[-1-x]
        return maj_class

    # 选择信息增益最大的特征
    best_i, gain_i = choose_best_feature(data, d_class)
    best_feature_label = tree_labels[best_i]
    decision_tree = {best_feature_label: {}}
    del (tree_labels[best_i])
    feature_value = set(data[:, best_i])

    # 分割子集
    for key in feature_value:
        sub_labels = tree_labels[:]
        new_list = np.array([np.delete(x, best_i) for x in data if x[best_i] == key])
        decision_tree[best_feature_label][key] = create_tree(new_list, sub_labels, d_class, name_class)

    return decision_tree


# 预测分类
def classfy(tree, tree_labels, test_sample):
    first_str = tree.keys()[0]
    second_dict = tree[first_str]
    feature_index = tree_labels.index(first_str)
    for key in second_dict.keys():
        if test_sample[feature_index] == key:
            if type(second_dict[key]).__name__ == 'dict':
                class_label = classfy(second_dict[key], tree_labels, test_sample)
            else:
                class_label = second_dict[key]
    return class_label


if __name__ == "__main__":
    data = pd.read_csv('iris.data', header=None)
    dummies = pd.get_dummies(data[4])
    data = np.array(data.ix[:, :3].join(dummies))
    labels = ['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width']
    name_class = ['Iris Setosa', 'Iris Versicolour', 'Iris Virginica']
    D = len(data)
    h_d = emp_entropy(data, 3, D)
    # 生成树
    my_tree = create_tree(data, labels, 3, name_class)

    #画树
    # treePlotter.createPlot(my_tree)
    testvec = data[10]

    print classfy(my_tree, labels, testvec)





















