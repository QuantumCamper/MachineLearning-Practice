#!/usr/bin/python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import matplotlib as mpl
from PIL import Image
from time import time
import matplotlib.pyplot as plt
import os


def save_image(im, i):
    im = 255 - im
    a = im.astype(np.uint8)
    out_path = '.\\hands_out'
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    Image.fromarray(a).save(out_path + ('\\%d.png' % i))


if __name__ == "__main__":
    classifier_type = 'rf'
    data = pd.read_csv('MNIST.train.csv', header=0)
    y = data['label'].values
    x = data.values[:, 1:]
    images = x.reshape(-1, 28, 28)
    y = y.ravel()

    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=0)
    images_train = x_train.reshape(-1, 28, 28)
    images_test = x_test.reshape(-1, 28, 28)

    mpl.rcParams['font.sans-serif'] = [u'SimHei']
    mpl.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(15, 9), facecolor='w')
    for index, image in enumerate(images_train[:16]):
        plt.subplot(4, 8, index + 1)
        plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
        plt.title(u'训练图片: %i' % y_train[index])
    for index, image in enumerate(images_test[:16]):
        plt.subplot(4, 8, index + 17)
        plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
        save_image(image.copy(), index)
        plt.title(u'测试图片: %i' % y_test[index])
    plt.tight_layout()
    plt.show()

    if classifier_type == 'svm':
        model = SVC(C=1000, kernel='rbf', gamma=1e-10)
        model.fit(x_train, y_train)
        y_hat = model.predict(x_test)
        print u'训练集准确率:', accuracy_score(y_train, model.predict(x_train))
        print u'测试集准确率:', accuracy_score(y_test, y_hat)

    elif classifier_type == 'rf':
        model = RandomForestClassifier(n_estimators=100, min_samples_split=2,
                                       min_impurity_split=1e-10, bootstrap=True, oob_score=True)
        model.fit(x_train, y_train)
        y_hat = model.predict(x_test)
        print u'训练集准确率:', accuracy_score(y_train, model.predict(x_train))
        print u'测试集准确率:', accuracy_score(y_test, y_hat)

    elif classifier_type == 'gbdt':
        t0 = time()
        model = GradientBoostingClassifier(learning_rate=0.08, n_estimators=100,
                                           min_samples_split=2, min_impurity_split=1e-10)
        model.fit(x_train, y_train)
        t1 = time()
        y_hat = model.predict(x_test)

        print u'训练集准确率:', accuracy_score(y_train, model.predict(x_train))
        print u'测试集准确率:', accuracy_score(y_test, y_hat)
        print u'训练时间：', t1 - t0

    err = (y_test != y_hat)
    err_images = images_test[err]
    err_y_hat = y_hat[err]
    err_y = y_test[err]
    print err_y_hat
    print err_y
    plt.figure(figsize=(10, 8), facecolor='w')
    for index, image in enumerate(err_images):
        if index >= 12:
            break
        plt.subplot(3, 4, index + 1)
        plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
        plt.title(u'错分为：%i，真实值：%i' % (err_y_hat[index], err_y[index]))
    plt.suptitle(u'数字图片手写体识别：分类器%s' % classifier_type, fontsize=18)
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.show()



