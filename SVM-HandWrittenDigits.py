#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from time import time

if __name__ == "__main__":
    data = np.loadtxt('optdigits.tra', delimiter=',')
    x_train, y_train = np.split(data, [-1, ], axis=1)
    images = x_train.reshape(-1, 8, 8)
    y_train = y_train.ravel().astype(np.int)

    data_test = np.loadtxt('optdigits.tes', delimiter=',')
    x_test, y_test = np.split(data_test, [-1, ], axis=1)
    images_test = x_test.reshape(-1, 8, 8)
    y_test = y_test.ravel().astype(np.int)

    mpl.rcParams['font.sans-serif'] = [u'SimHei']
    mpl.rcParams['axes.unicode_minus'] = False
    plt.figure(facecolor='w')
    for i, image in enumerate(images[:16]):
        plt.subplot(4, 8, i+1)
        plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
        plt.title(u'训练图片： %d' % y_train[i])
    for i, image in enumerate(images_test[:16]):
        plt.subplot(4, 8, i+17)
        plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
        plt.title(u'训练图片： %d' % y_test[i])
    plt.tight_layout()
    plt.show()

    model = SVC(C=10, kernel='rbf', gamma=0.001)
    print u'time begin'
    t0 = time()
    model.fit(x_train, y_train)
    t1 = time()
    t = t1 - t0
    print 'time end'
    print u'训练时间:%d分钟,%.3f秒' % (int(t/60), (t - 60*(t/60)))

    y_pre = model.predict(x_test)
    print u'分类结果', classification_report(y_test, y_pre)
    print u'训练集准确率:', accuracy_score(y_train, model.predict(x_train))
    print u'测试集准确率:', accuracy_score(y_test, y_pre)

    error_images = images_test[y_pre != y_test]
    error_y_pre = y_pre[y_pre != y_test]
    error_y_test = y_test[y_pre != y_test]
    print error_y_pre
    print error_y_test
    plt.figure(facecolor='w')
    for i, image in enumerate(error_images):
        if i >= 12:
            break
        plt.subplot(3, 4, i+1)
        plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
        plt.title(u'误分为: %d， 真实值: %d' % (error_y_pre[i], error_y_test[i]))

    plt.tight_layout()
    plt.show()







