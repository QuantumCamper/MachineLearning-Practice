# !/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, SGDRegressor, RidgeCV, LassoCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import matplotlib as mpl
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # 数据导入
    file_data = pd.read_csv('housing.data', header=None, sep='\s+')
    x = np.array(file_data.ix[:, :12])
    y = np.array(file_data.ix[:, 13])
    # 数据标准化
    standard = StandardScaler()
    x = standard.fit_transform(x)
    poly = PolynomialFeatures(2)# 2阶
    # # poly = PolynomialFeatures(3)#3阶
    x = poly.fit_transform(x)
    # 训练集/测试集选择
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=0)

    # 训练模型
    # model = LinearRegression(fit_intercept=False)
    # model = SGDRegressor(penalty=None, fit_intercept=True, learning_rate='constant', eta0=0.00001, n_iter=10000)
    model = RidgeCV()
    y_hat = model.predict(x_test)
    R2 = model.score(x_test, y_test)
    print R2
    #结果展示1:真实值/预测值
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

