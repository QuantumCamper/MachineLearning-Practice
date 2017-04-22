# !/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import pairwise_distances_argmin

mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False

if __name__ == '__main__':
    np.random.seed(0)
    mu1_fact = (0, 0, 0)
    cov1_fact = np.diag((1, 2, 3))
    data1 = np.random.multivariate_normal(mu1_fact, cov1_fact, 400)

    mu2_fact = (2, 2, 1)
    cov2_fact = np.array(((1, 1, 3), (1, 2, 1), (0, 0, 1)))
    data2 = np.random.multivariate_normal(mu2_fact, cov2_fact, 100)
    data = np.vstack((data1, data2))
    y = np.array([True] * 400 + [False] * 100)

    #sklearn EM
    # g = GaussianMixture(n_components=2, covariance_type='full', tol=1e-6, max_iter=100)
    # g.fit(data)
    # print '类别概率:\t', g.weights_
    # print '均值:\t', g.means_
    # print '方差:\n', g.covariances_
    # mul1, mul2 = g.means_
    # sigma1, sigma2 = g.covariances_

    # EM practice
    n, d = data.shape
    mul1 = data.min(axis=0)
    mul2 = data.max(axis=0)
    sigma1 = np.identity(d)
    sigma2 = np.identity(d)
    pi = 0.5

    for i in range(50):
        # E step
        func1 = multivariate_normal(mul1, sigma1)
        func2 = multivariate_normal(mul2, sigma2)
        part1 = pi * func1.pdf(data)
        part2 = (1 - pi) * func2.pdf(data)
        Q1 = part1 / (part1 + part2)

        # M step
        mul1 = np.dot(Q1, data) / np.sum(Q1)
        mul2 = np.dot((1 - Q1), data) / np.sum(1 - Q1)
        sigma1 = np.dot(Q1 * (data - mul1).T, (data - mul1)) / np.sum(Q1)
        sigma2 = np.dot((1 - Q1) * (data - mul2).T, (data - mul2)) / np.sum(1 - Q1)
        pi = np.sum(Q1) / n
        print i, "均值为：", mul1, mul2

    print '类别概率:\t', pi
    print '均值:\t', mul1, mul2
    print '方差:\n', sigma1, '\n\n', sigma2, '\n'

    # 预测分类
    norm1 = multivariate_normal(mul1, sigma1)
    norm2 = multivariate_normal(mul2, sigma2)
    tau1 = norm1.pdf(data)
    tau2 = norm2.pdf(data)

    fig = plt.figure(figsize=(13, 7), facecolor='w')
    ax = fig.add_subplot(121, projection='3d')
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c='b', s=30, marker='o', depthshade=True)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(u'原始数据', fontsize=18)
    ax = fig.add_subplot(122, projection='3d')
    order = pairwise_distances_argmin([mu1_fact, mu2_fact], [mul1, mul2], metric='euclidean')
    if order[0] == 0:
        c1 = tau1 > tau2
    else:
        c1 = tau1 < tau2
    c2 = ~c1
    acc = np.mean(y == c1)
    print u'准确率：%.2f%%' % (100 * acc)
    ax.scatter(data[c1, 0], data[c1, 1], data[c1, 2], c='r', s=30, marker='o', depthshade=True)
    ax.scatter(data[c2, 0], data[c2, 1], data[c2, 2], c='g', s=30, marker='^', depthshade=True)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(u'EM算法分类', fontsize=18)
    plt.suptitle(u'EM算法的实现', fontsize=21)
    plt.subplots_adjust(top=0.90)
    plt.tight_layout()
    plt.show()










