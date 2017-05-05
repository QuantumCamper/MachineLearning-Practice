# !user/bin/python
# -*- coding:utf-8 -*-

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


# 最大似然估计参数
def max_likelihood_estimate(x_est, y_est):
    y_class = list(set(y_est))
    y_prior = [float(list(y_est).count(c)) / len(y_est) for c in y_class]
    mul_sigma2 = {}
    for i in range(len(x_est[0])):
        for j in y_class:
            index = np.where(y_est == j)[0]
            x_y = [x_est[a, i] for a in index]
            mul = np.sum(x_y) / len(x_y)
            sigma_2 = np.sum((x_y - mul) * (x_y - mul)) / len(x_y)
            mul_sigma2[(i, j)] = (mul, sigma_2)
    return mul_sigma2, y_prior


# 高斯概率计算
def gaussian(x_pre, mul, sigma2):
    return 1.0/np.sqrt(2 * np.pi * sigma2) * np.exp(-pow((x_pre - mul), 2)/(2 * sigma2))


# 计算朴素贝叶斯最大后验概率
def naive_bayes(x_pre, y_class, mul_sigma2, y_prior):
    y_probability = []
    for label in y_class:
        prob_label = 1
        for k in range(len(x_pre)):
            mul = mul_sigma2[(k, label)][0]
            sigma2 = mul_sigma2[(k, label)][1]
            max_estimate = gaussian(x_pre[k], mul, sigma2)
            prob_label *= max_estimate
        p = y_prior[label] * prob_label
        y_probability.append(p)
    return y_probability


# 预测值
def predict(x_data, y_data, class_labels, mul_sigma2, y_prior):
    y_hat = []
    n_right = 0
    for j in range(len(x_data)):
        prob = naive_bayes(x_data[j], class_labels, mul_sigma2, y_prior)
        n = np.argmax(prob)
        y_hat.append(class_labels[n])
        if class_labels[n] == y_data[j]:
            n_right += 1

    precision = float(n_right) / len(y_data)
    return y_hat, precision

if __name__ == "__main__":
    data = pd.read_csv('iris.data', header=None)
    x, y = data[range(0, 4)], data[4]
    y = pd.Categorical(values=y).codes
    x, y = np.array(x), np.array(y)
    # y = pd.get_dummies(y, prefix='class_')
    # it = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
    # y = y.map(it)
    feature_names = u'花萼长度', u'花萼宽度', u'花瓣长度', u'花瓣宽度'
    features = [0, 1]
    x = x[:, features]
    labels = list(set(y))
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=0)
    string = 'gnb'

    if string == 'my_self':
        mul_sigma, y_prob = max_likelihood_estimate(x_train, y_train)
        n_right_train = 0
        n_right_test = 0
        y_pre, pre = predict(x_train, y_train, labels, mul_sigma, y_prob)
        y_pre_test, pre_test = predict(x_test, y_test, labels, mul_sigma, y_prob)
        print "训练集准确率为：", pre
        print "测试集准确率为：", pre_test
    else:
        priors = np.array([1, 2, 1], dtype=float)
        priors /= priors.sum()
        gnb = Pipeline([('standard', StandardScaler()),
                        ('poly', PolynomialFeatures(degree=1)),
                        ('gnb', GaussianNB(priors=priors))])
        gnb.fit(x_train, y_train.ravel())
        y_pre = gnb.predict(x_train)
        print "训练集准确率为：%.2f%%" % (accuracy_score(y_train, y_pre) * 100)
        y_pre_test = gnb.predict(x_test)
        print "测试集准确率为：%.2f%%" % (accuracy_score(y_test, y_pre_test) * 100)

    # 画图
    N, M = 500, 500
    x1_min, x2_min = x_train[:, features[0]].min(), x_train[:, features[1]].min()
    x1_max, x2_max = x_train[:, features[0]].max(), x_train[:, features[1]].max()
    t1 = np.linspace(x1_min, x1_max, N)
    t2 = np.linspace(x2_min, x2_max, M)
    x1, x2 = np.meshgrid(t1, t2)
    x_grid = np.stack((x1.flat, x2.flat), axis=1)


    mpl.rcParams['font.sans-serif'] = [u'simHei']
    mpl.rcParams['axes.unicode_minus'] = False
    cm_light = mpl.colors.ListedColormap(['#77E0A0', '#FF8080', '#A0A0FF'])
    cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])
    y_grid_hat = gnb.predict(x_grid)
    y_grid_hat = y_grid_hat.reshape(x1.shape)
    plt.pcolormesh(x1, x2, y_grid_hat, cmap=cm_light)  # 预测值的显示

    plt.scatter(x_train[:, features[0]], x_train[:, features[1]], c=y_train, edgecolors='k', s=50, cmap=cm_dark)
    plt.scatter(x_test[:, features[0]], x_test[:, features[1]], c=y_test, marker='^', edgecolors='k', s=120,
                cmap=cm_dark)

    plt.xlabel(feature_names[features[0]], fontsize=13)
    plt.ylabel(feature_names[features[1]], fontsize=13)
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.title(u'GaussianNB对鸢尾花数据的分类结果', fontsize=18)
    plt.grid(True)
    plt.show()























