# !user/bin/python
# -*- coding:utf-8 -*-

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
import matplotlib as mpl
import matplotlib.pyplot as plt


def nonempty(s):
    return s != ''

if __name__ == "__main__":
    file_in = open('housing.data')
    data = []
    for line in file_in:
        line = filter(nonempty, line.strip().split(' '))
        data.append(map(float, line))
    x = np.array([t[:13] for t in data])
    y = np.array([t[13] for t in data])
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=0)

    regressor = DecisionTreeRegressor(random_state=0)

    # 选择最优交叉验证次数
    # max_depths = 10
    # scores = list()
    # std_scores = list()
    # n_folds = range(2, 20)
    # for n_fold in n_folds:
    #     regressor.max_depth = max_depths
    #     this_score = cross_val_score(regressor, x_train, y_train, cv=n_fold)
    #     scores.append(np.mean(this_score))
    #     std_scores.append(np.std(this_score))
    # scores, std_scores = np.array(scores), np.array(std_scores)
    #
    # plt.figure(figsize=(10, 8), facecolor='w')
    # plt.plot(n_folds, scores, color='g', lw=1.8, label='scores')
    # std_error = np.array([score/np.sqrt(n) for score, n in zip(std_scores, n_folds)])
    # plt.plot(n_folds, scores + std_error, 'b--')
    # plt.plot(n_folds, scores - std_error, 'b--')
    # plt.fill_between(n_folds, scores + std_error, scores - std_error, alpha=0.2)
    # plt.ylabel('scores +/- std_error')
    # plt.xlabel('n_folds')
    # plt.axhline(np.max(scores), linestyle='--', color='.5')
    # plt.xlim([n_folds[0], n_folds[-1]])
    # plt.show()

    # 选择最优深度
    # max_depths = range(2, 20)
    # scores = list()
    # std_scores = list()
    # n_folds = 9
    # for depth in max_depths:
    #     regressor.max_depth = depth
    #     this_score = cross_val_score(regressor, x_train, y_train, cv=n_folds)
    #     scores.append(np.mean(this_score))
    #     std_scores.append(np.std(this_score))
    # scores, std_scores = np.array(scores), np.array(std_scores)
    #
    # plt.figure(figsize=(10, 8), facecolor='w')
    # plt.plot(max_depths, scores, color='g', lw=1.8, label='scores')
    # std_error = std_scores/np.sqrt(n_folds)
    # plt.plot(max_depths, scores + std_error, 'b--')
    # plt.plot(max_depths, scores - std_error, 'b--')
    # plt.fill_between(max_depths, scores + std_error, scores - std_error, alpha=0.2)
    # plt.ylabel('scores +/- std_error')
    # plt.xlabel('max_depths')
    # plt.axhline(np.max(scores), linestyle='--', color='.5')
    # plt.xlim([max_depths[0], max_depths[-1]])
    # plt.show()

    # 选择最优叶子节点数
    # max_leaf = range(2, 20)
    # scores = list()
    # std_scores = list()
    # n_folds = 9
    # for leaf in max_leaf:
    #     regressor.max_depth = 14
    #     regressor.max_leaf_nodes = leaf
    #     this_score = cross_val_score(regressor, x_train, y_train, cv=n_folds)
    #     scores.append(np.mean(this_score))
    #     std_scores.append(np.std(this_score))
    # scores, std_scores = np.array(scores), np.array(std_scores)
    #
    # plt.figure(figsize=(10, 8), facecolor='w')
    # plt.plot(max_leaf, scores, color='g', lw=1.8, label='scores')
    # std_error = std_scores/np.sqrt(n_folds)
    # plt.plot(max_leaf, scores + std_error, 'b--')
    # plt.plot(max_leaf, scores - std_error, 'b--')
    # plt.fill_between(max_leaf, scores + std_error, scores - std_error, alpha=0.2)
    # plt.ylabel('scores +/- std_error')
    # plt.xlabel('max_leaf')
    # plt.axhline(np.max(scores), linestyle='--', color='.5')
    # plt.xlim([max_leaf[0], max_leaf[-1]])
    # plt.show()

    # 模型计算
    regressor.max_depth = 14
    # regressor.max_leaf_nodes = 14
    model = regressor.fit(x_train, y_train)
    y_pre = model.predict(x_test)
    score = model.score(x_test, y_test)
    print score
    mpl.rcParams['font.sans-serif'] = [u'SimHei']
    mpl.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(10, 8), facecolor='w')
    t = range(len(x_test))
    plt.plot(t, y_test, color='g', lw=1.8, label=u'真实值')
    plt.plot(t, y_pre, color='r', lw=1.8, label=u'预测值')
    plt.legend(loc='best')
    plt.show()

