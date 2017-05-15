#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np


def load_params():
    f1 = open('train_pi.txt')
    PI = [float(pi) for pi in f1.read().strip().split(' ')]
    f1.close()
    f2 = open('train_A.txt')
    A = [[float(w) for w in line.strip().split(' ')] for line in f2]
    f2.close()
    f3 = open('train_B.txt')
    B = [[float(b) for b in line.strip().split(' ')] for line in f3]
    return PI, A, B


# 维特比算法
def viterbi(sentence, PI, A, B, key):
    n = len(sentence)
    delta = np.zeros((n, key))
    phi = np.zeros((n, key))
    decode = [-1] * n
    for i in range(key):
        delta[0][i] = PI[i] + B[i][ord(sentence[0])]
    for t in range(1, n):
        for k in range(key):
            result = [delta[t-1][j] + A[j][k] for j in range(key)]
            index = np.argmax(result)
            delta[t][k] = result[index] + B[k][ord(sentence[t])]
            phi[t][k] = index
    q = np.argmax([q_T for q_T in delta[n-1]])
    decode[n-1] = q
    for t in range(1, n):
        q = int(phi[n-t][q])
        decode[n-1-t] = q
    return decode


# segment
def segment(sentence, decode):
    n = len(sentence)
    i = 0
    while i < n:
        if decode[i] == 0 or decode[i] == 1:
            j = i+1
            while j < n:
                if decode[j] == 2:
                    break
                j += 1
            print sentence[i:j+1], '|',
            i = j + 1
        elif decode[i] == 2 or decode[i] == 3:
            print sentence[i:i+1], '|',
            i += 1
        else:
            print 'Error: 第%d个词发生错误'% i, decode[i]
            i += 1


if __name__ == "__main__":
    pi, a, b = load_params()
    key_num = 4
    f = open('novel.txt')
    data = f.read()[3:].decode('utf-8')
    f.close()
    # 维特比算法预测
    seg = viterbi(data, pi, a, b, key_num)
    print seg
    # 打印输出
    segment(data, seg)





