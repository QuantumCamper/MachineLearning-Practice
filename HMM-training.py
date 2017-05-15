#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import ChineseProcess


# normalize
def log_normalize(nm_data):
    sum_data = np.sum(nm_data)
    out_data = [np.log(d) - np.log(sum_data) if d != 0 else float(-2**31) for d in nm_data]
    return out_data


# 最大似然估计HMM参数
def mle(train_data):
    train_data = train_data.split('  ')
    num_data = len(train_data)
    a = np.zeros((key, key))
    pi = np.zeros(key)
    b = np.zeros((key, 65536))
    last_q = 2
    first_b = 1
    progress = [num_data*i/10 for i in range(1, 10)]
    for index, tokens in enumerate(train_data):
        if index in progress:
            print u'已完成：%.2f%%' % (100*(float(index)/num_data))
        tokens = tokens.strip()
        n = len(tokens)
        if n <= 0:
            continue
        if n == 1:
            a[last_q][3] += 1
            b[3][ord(tokens[0])] += 1
            if first_b == 1:
                pi[3] += 1
            last_q = 3
            continue
        if first_b == 1:
            pi[0] += 1
        a[last_q][0] += 1
        last_q = 2
        if n == 2:
            a[0][2] += 1
        else:
            a[0][1] += 1
            a[1][1] += (n-3)
            a[1][2] += 1
        b[0][ord(tokens[0])] += 1
        b[2][ord(tokens[n-1])] += 1
        for num in range(1, n-1):
            b[1][ord(tokens[num])] += 1
        if ChineseProcess.is_other(tokens):
            first_b = 1
        else:
            first_b = 0
    pi = log_normalize(pi)
    for normal in range(key):
        a[normal] = log_normalize(a[normal])
        b[normal] = log_normalize(b[normal])
    return pi, a, b


# 字符串写
def save_params(f, p):
    for num in p:
        f.write(str(num))
        f.write(' ')
    f.write('\n')

if __name__ == "__main__":
    f1 = open(".\\pku_training.utf8")
    data = f1.read()[3:].decode('utf-8')
    f1.close()
    key = 4
    pi_est, a_est, b_est = mle(data)

    # 保存参数
    f_pi = open('train_pi.txt', 'w')
    save_params(f_pi, pi_est)
    f_a = open('train_A.txt', 'w')
    f_b = open('train_B.txt', 'w')
    for k in range(key):
        save_params(f_a, a_est[k])
        save_params(f_b, b_est[k])
    f_pi.close()
    f_a.close()
    f_b.close()





