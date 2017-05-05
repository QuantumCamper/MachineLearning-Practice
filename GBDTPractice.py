# !user/bin/python
# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from matplotlib.pyplot import rcParams


def model_fit(alg, train_x, train_y, test_x, test_y, perform_cv=True, print_feature_importance=True, cv_folds=5):
    # fit 函数
    alg.fit(train_x, train_y)

    # 预测
    y_pre = alg.predict(test_x)
    r2 = alg.score(x_test, y_test)

    # 交叉验证
    if perform_cv:
        cv_score = cross_val_score(alg, train_x, train_y, scoring='r2', cv=cv_folds)

    # 输出
    print "\nModel Report"
    print "MSE: %.4f" % mean_squared_error(test_y, y_pre)
    print "R2: %.4f" % r2

    if perform_cv:
        print "CV score: Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g" % \
              (np.mean(cv_score), np.std(cv_score), np.min(cv_score), np.max(cv_score))

    # 特征重要度
    if print_feature_importance:
        feat_imp = pd.Series(alg.feature_importances_, np.arange(len(train_x[0]))).sort_values(ascending=False)
        feat_imp.plot(kind='bar', title='Feature Importance')
        plt.ylabel('Feature Importance Score')


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
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=10)

    # 基准模型
    # gbdt = GradientBoostingRegressor(random_state=10)
    # model_fit(gbdt, x_train, y_train, x_test, y_test)

    # 选择最优树数目
    # params_test1 = {'n_estimators': range(20, 350, 10)}
    # grid_search1 = GridSearchCV(estimator=GradientBoostingRegressor(learning_rate=0.15,
    #                                                                 min_samples_split=3, min_samples_leaf=5,
    #                                                                 max_depth=5, max_features='sqrt', subsample=0.8,
    #                                                                 random_state=10), param_grid=params_test1,
    #                                                                 scoring='r2', n_jobs=4,
    #                                                                 iid=False, cv=5)
    # grid_search1.fit(x_train, y_train)
    # print "mean_test_score:", grid_search1.cv_results_['mean_test_score']
    # print "std_test_score:", grid_search1.cv_results_['std_test_score']
    # print grid_search1.best_params_, grid_search1.best_score_

    # 选择 max_depth and min_samples_split
    # params_test2 = {'max_depth': range(3, 20, 2), 'min_samples_split': range(2, 10, 2)}
    # grid_search1 = GridSearchCV(estimator=GradientBoostingRegressor(learning_rate=0.15, n_estimators=150,
    #                                                                 min_samples_leaf=5,
    #                                                                 max_features='sqrt', subsample=0.8,
    #                                                                 random_state=10), param_grid=params_test2,
    #                                                                 scoring='r2', n_jobs=4, iid=False, cv=5)
    # grid_search1.fit(x_train, y_train)
    # print "mean_test_score:", grid_search1.cv_results_['mean_test_score']
    # print "std_test_score:", grid_search1.cv_results_['std_test_score']
    # print grid_search1.best_params_, grid_search1.best_score_

    # 选择 min_samples_leaf
    # params_test2 = {'min_samples_leaf': range(2, 10)}
    # grid_search1 = GridSearchCV(estimator=GradientBoostingRegressor(learning_rate=0.15, n_estimators=150,
    #                                                                 min_samples_split=2, max_depth=5,
    #                                                                 max_features='sqrt', subsample=0.8,
    #                                                                 random_state=0), param_grid=params_test2,
    #                                                                 scoring='r2', n_jobs=4, iid=False, cv=5)
    # grid_search1.fit(x_train, y_train)
    # print "mean_test_score:", grid_search1.cv_results_['mean_test_score']
    # print "std_test_score:", grid_search1.cv_results_['std_test_score']
    # print grid_search1.best_params_, grid_search1.best_score_
    #
    # # 验证下结果
    # model_fit(grid_search1.best_estimator_, x_train, y_train, x_test, y_test)

    # 选择 max_features
    # params_test2 = {'max_features': range(3, 13)}
    # grid_search1 = GridSearchCV(estimator=GradientBoostingRegressor(learning_rate=0.15, n_estimators=150,
    #                                                                 min_samples_split=2, max_depth=5,
    #                                                                 min_samples_leaf=2, subsample=0.8,
    #                                                                 random_state=0), param_grid=params_test2,
    #                                                                 scoring='r2', n_jobs=4, iid=False, cv=5)
    # grid_search1.fit(x_train, y_train)
    # print "mean_test_score:", grid_search1.cv_results_['mean_test_score']
    # print "std_test_score:", grid_search1.cv_results_['std_test_score']
    # print grid_search1.best_params_, grid_search1.best_score_

    # 选择 max_features
    # params_test2 = {'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9]}
    # grid_search1 = GridSearchCV(estimator=GradientBoostingRegressor(learning_rate=0.15, n_estimators=150,
    #                                                                 min_samples_split=2, max_depth=5,
    #                                                                 min_samples_leaf=2, max_features='sqrt',
    #                                                                 random_state=0), param_grid=params_test2,
    #                                                                 scoring='r2', n_jobs=4, iid=False, cv=5)
    # grid_search1.fit(x_train, y_train)
    # print "mean_test_score:", grid_search1.cv_results_['mean_test_score']
    # print "std_test_score:", grid_search1.cv_results_['std_test_score']
    # print grid_search1.best_params_, grid_search1.best_score_

    # 选择 learning_rate, n_estimators

    gbdt2 = GradientBoostingRegressor(learning_rate=0.0025, n_estimators=9000,
                                      min_samples_split=2, max_depth=5,subsample=0.8, min_samples_leaf=2,
                                      max_features='sqrt', random_state=0)


    # 验证下结果
    model_fit(gbdt2, x_train, y_train, x_test, y_test)



