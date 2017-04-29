# !/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
from numpy import linalg
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
import matplotlib as mpl
import matplotlib.pyplot as plt


# Batch gradient descent
def gradient_descent(x, y, x_test, y_test, alpha=0.000001, iter=10000, batch=200):
    m, n = x.shape
    weights = np.ones(n)
    error_mean = []
    error_test_mean = []

    for i in range(iter):
        err_iter = np.dot(x, np.transpose(weights)) - y
        weights = weights - alpha * np.dot(np.transpose(x), err_iter)

        if i % batch == 0:
            error = np.dot(x, np.transpose(weights)) - y
            error_sum = np.sum(np.power(error, 2)) / m
            error_mean.append(error_sum)
            error_test = test_result(x_test, y_test, weights)
            error_test_mean.append(error_test)

    return weights, error_mean, error_test_mean


# Stochastic gradient descent
def stochastic_gradient_descent(x, y, x_test, y_test, alpha=0.0001, iter=10000, batch=200):
    m, n = x.shape
    weights = np.ones(n)
    error_mean = []
    error_test_mean = []

    for i in range(iter):
        index = np.random.randint(0, m)
        err_iter = np.dot(x[index], np.transpose(weights)) - y[index]
        weights = weights - alpha * err_iter * x[index]

        if i % batch == 0:
            error = np.dot(x, np.transpose(weights)) - y
            error_sum = np.sum(np.power(error, 2)) / m
            error_mean.append(error_sum)
            error_test = test_result(x_test, y_test, weights)
            error_test_mean.append(error_test)

    return weights, error_mean, error_test_mean


# minibatch Stochastic gradient descent
def minibatch_stochastic_gradient_descent(x, y, x_test, y_test, alpha=0.0001,
                                          iter=10000, batch=200, minibatch=20):
    m, n = x.shape
    weights = np.ones(n)
    error_mean = []
    error_test_mean = []

    for i in range(iter):
        for j in range(minibatch):
            index = np.random.randint(0, m)
            err_iter = np.dot(x[index], np.transpose(weights)) - y[index]
            weights = weights - alpha * err_iter * x[index]

        if i % batch == 0:
            error = np.dot(x, np.transpose(weights)) - y
            error_sum = np.sum(np.power(error, 2)) / m
            error_mean.append(error_sum)
            error_test = test_result(x_test, y_test, weights)
            error_test_mean.append(error_test)

    return weights, error_mean, error_test_mean


# test result
def test_result(x_test, y_test, weight):
    m_test = len(x_test)
    error_test = np.dot(x_test, np.transpose(weight)) - y_test
    error_mean_pre = np.sum(np.power(error_test, 2)) / m_test
    return error_mean_pre


# normal equations
def stand_regression(x, y, beta):
    x_matrix = np.mat(x)
    y_matrix = np.mat(y).T
    xTx = x_matrix.T * x_matrix
    beta_unite = np.mat(beta * np.eye(xTx.shape[0], xTx.shape[1]))

    if linalg.det(xTx) == 0:
        print "This Matrix is singular, can not inverse"
        theta = (xTx + beta_unite).I * (x_matrix.T * y_matrix)
        return theta
    else:
        theta = xTx.I * (x_matrix.T * y_matrix)
        return theta


# minibatch Stochastic gradient descent + Ridge
def stochastic_ridge(x, y, x_test, y_test, alpha=0.0001,
                                          iter=10000, batch=200, minibatch=20, beta=1.0):
    m, n = x.shape
    weights = np.ones(n)
    error_mean = []
    error_test_mean = []

    for i in range(iter):
        for j in range(minibatch):
            index = np.random.randint(0, m)
            err_iter = np.dot(x[index], np.transpose(weights)) - y[index]
            weights = weights - alpha * err_iter * x[index] + beta * weights

        if i % batch == 0:
            error = np.dot(x, np.transpose(weights)) - y
            error_sum = np.sum(np.power(error, 2)) / m
            error_mean.append(error_sum)
            error_test = test_result(x_test, y_test, weights)
            error_test_mean.append(error_test)

    return weights, error_mean, error_test_mean


# minibatch Stochastic gradient descent + Lasso
def stochastic_lasso(x, y, x_test, y_test, alpha=0.0001,
                                          iter=10000, batch=200, minibatch=20, beta=1.0):
    m, n = x.shape
    weights = np.ones(n)
    error_mean = []
    error_test_mean = []

    for i in range(iter):
        for j in range(minibatch):
            index = np.random.randint(0, m)
            err_iter = np.dot(x[index], np.transpose(weights)) - y[index]
            l1_norm = np.array(lasso_derivative(weights))
            weights = weights - alpha * err_iter * x[index] + beta * l1_norm

        if i % batch == 0:
            error = np.dot(x, np.transpose(weights)) - y
            error_sum = np.sum(np.power(error, 2)) / m
            error_mean.append(error_sum)
            error_test = test_result(x_test, y_test, weights)
            error_test_mean.append(error_test)

    return weights, error_mean, error_test_mean


# derivative of L1-norm
def lasso_derivative(w, alpha=10e6):
    w_lasso = []
    for i in range(len(w)):
        w_la = 1.0/(1 + np.exp(- (alpha * w[i]))) - 1.0/(1 + np.exp(alpha * w[i]))
        w_lasso.append(w_la)
    return w_lasso

if __name__ == "__main__":

    # 数据导入
    file_data = pd.read_csv('housing.data', header=None, sep='\s+')
    x = np.array(file_data.ix[:, :12])
    y = np.array(file_data.ix[:, 13])

    # 添加截距
    new_col = np.ones((len(x), 1))
    x = np.column_stack((x, new_col))

    # 数据标准化
    standard = StandardScaler()
    x = standard.fit_transform(x)
    poly = PolynomialFeatures(2)# 2阶
    # poly = PolynomialFeatures(3)#3阶
    x = poly.fit_transform(x)

    # 训练集/测试集选择
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=0)

    # 训练模型
    # # minibatch SGD
    # weights, error_mean, error_test_mean = minibatch_stochastic_gradient_descent(x_train, y_train,
    #                                                                              x_test, y_test,
    #                                                                              alpha=0.00002, iter=100000)  # 2阶
    # # SGD
    # weights, error_mean, error_test_mean = stochastic_gradient_descent(x_train, y_train,
    #                                                                    x_test, y_test,
    #                                                                    alpha=0.000002, iter=1000000)#3阶
    # normal equations
    weights = stand_regression(x_train, y_train,beta=7)
    weights = weights.T
    #
    # # SGD + Ridge
    # weights, error_mean, error_test_mean = stochastic_ridge(x_train, y_train, x_test, y_test,
    #                                                             alpha=0.0002, iter=10000, beta=0.0000003)  # 2阶

    # SGD + Lasso
    # weights, error_mean, error_test_mean = stochastic_lasso(x_train, y_train, x_test, y_test,
    #                                                         alpha=0.0002, iter=10000, beta=0.0000003)
    y_hat = np.dot(x_test, np.transpose(weights))
    R2 = r2_score(y_test, y_hat)
    print R2


    # 结果展示1:真实值/预测值
    mpl.rcParams['font.sans-serif'] = [u'simHei']
    mpl.rcParams['axes.unicode_minus'] = False
    t1 = np.arange(len(y_hat))
    plt.figure(figsize=(10, 8), facecolor='w')
    plt.subplot(1, 2, 1)
    plt.plot(t1, y_test, 'r-', lw=1.5, label=u'真实值')
    plt.plot(t1, y_hat, 'g-', lw=1.5, label=u'预测值')
    plt.legend(loc='best')
    plt.title(u'波士顿房价预测', fontsize=18)
    plt.xlabel(u'样本编号', fontsize=18)
    plt.ylabel(u'房屋价格', fontsize=18)
    plt.show()
    # 结果展示2:均方误差/准确率
    # t2 = np.arange(len(error_mean))
    # plt.subplot(1, 2, 2)
    # plt.plot(t2, error_mean, 'r-', lw=1.5, label=u'训练误差')
    # plt.plot(t2, error_test_mean, 'g-', lw=1.5, label=u'测试误差')
    # plt.legend(loc='best')
    # plt.title(u'均方误差/准确率', fontsize=18)
    # plt.xlabel(u'迭代次数', fontsize=18)
    # plt.ylabel(u'误差', fontsize=18)
    # plt.tight_layout()
    # plt.show()













