#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import pandas as pd
import jieba
import re
from sklearn.feature_extraction import text
from sklearn.cluster import k_means, MiniBatchKMeans, affinity_propagation, spectral_clustering, \
    AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph
from sklearn.covariance import empirical_covariance
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import codecs

def remain_year(x):
    return int(x[:4])

def remove_stopwords():
    stopwords_expend = {}.fromkeys([u'一种', u'设备', u'的', u'装置', u'方法', u'用于', u'基于'])
    with open('patent_title', 'w') as f:
        for item in data_title:
            seg_list = jieba.cut(item)
            words = []
            for word in seg_list:
                if word not in stopwords_expend:
                    words.append(word)
            words = ' '.join(words)
            print(words)
            f.write(words.encode('utf-8') + '\n')

def kmeans():
    f = open('patent_title').readlines()
    model_tfidf = text.TfidfVectorizer('content')
    title_vector = model_tfidf.fit_transform(f)
    model_cluster = k_means(title_vector, n_clusters=20, random_state=0)
    labels = model_cluster[1]
    data['cluster_label'] = labels
    data_cluster = data[[u'申请号', u'名称', 'cluster_label']]
    data_cluster.to_csv('data_cluster.xls', encoding='utf-8-sig')


def minibatch_kmeans():
    f = open('patent_title').readlines()
    model_tfidf = text.TfidfVectorizer('content')
    title_vector = model_tfidf.fit_transform(f)
    model_cluster = MiniBatchKMeans(n_clusters=20, random_state=0)
    model_cluster.fit_transform(title_vector)
    labels = model_cluster.labels_
    data['cluster_label'] = labels
    data_cluster = data[[u'申请号', u'名称', 'cluster_label']]
    data_cluster.to_csv('data_cluster_minibatch.xls', encoding='utf-8-sig')

def spectral():
    f = open('patent_title').readlines()
    model_tfidf = text.TfidfVectorizer('content')
    title_vector = model_tfidf.fit_transform(f)
    title_vector = kneighbors_graph(title_vector, n_neighbors=10)
    title_vector = title_vector.toarray()

    labels = spectral_clustering(title_vector, n_clusters=100)
    print(len(labels))
    data['cluster_label'] = labels
    data_cluster = data[[u'申请号', u'名称', 'cluster_label']]
    data_cluster.to_csv('data_cluster_ap.xls', encoding='utf-8-sig')

def agnes():
    f = open('patent_title').readlines()
    model_tfidf = text.TfidfVectorizer('content')
    title_vector = model_tfidf.fit_transform(f)
    title_vector = kneighbors_graph(title_vector, n_neighbors=10)
    title_vector = title_vector.toarray()

    model = AgglomerativeClustering(n_clusters=100)
    model.fit(title_vector)
    labels = model.labels_
    data['cluster_label'] = labels
    data_cluster = data[[u'申请号', u'名称', 'cluster_label']]
    data_cluster.to_csv('data_cluster_ap.xls', encoding='utf-8-sig')



if __name__ == "__main__":
    data = pd.read_excel(u'国内初检专利列表.xls', header=0)
    # 去重
    data.drop_duplicates([u'申请号'])
    data.drop([u'国省代码', u'国际公布', u'进入国家日期'], axis=1, inplace=True)
    public_data = data[u'公开（公告）日'].map(remain_year)
    data_title = data[u'名称']

    # pattern = ur'球[\u4e00-\u9fa5]+?钛'
    # pattern = re.compile(pattern)
    # for item in data[u'摘要']:
    #     find_list = re.findall(pattern, item)
    #     if find_list:
    #         for j in find_list:
    #             print(j,)
    agnes()
















    
