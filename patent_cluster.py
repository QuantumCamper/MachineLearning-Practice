
# coding: utf-8

# In[269]:

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import pandas as pd
import jieba
import re
import os
from sklearn.feature_extraction import text
from sklearn.cluster import k_means, MiniBatchKMeans, affinity_propagation, spectral_clustering,     AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph
from sklearn.covariance import empirical_covariance
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import codecs
import csv


# In[270]:

# 遍历指定路径下所有文件
def view_eachfile(filepath):
    pathdir = os.listdir(filepath)
    child = []
    for path in pathdir:
        child.append(os.path.join(filepath, path))
    return child
    
def remain(x, num, label=0):
    if x == 0:
        pass
    elif (label == 1):
        return x[:num]
    else:
        return int(x[:num])
  

#判断中文
def zhpattern(context):
    zhPattern = re.compile(u'[\u4e00-\u9fa5]+')
    match = zhPattern.search(context)
    if match:
        return True
    else:
        return False

# 词频统计
def words_count(targetlist):
    
    # 去除停止词
    stopwords_expend = {}.fromkeys([u'一种', u'设备', u'的', u'装置', u'方法', u'用于', u'基于'])
    with open('patent_segment.txt', 'w') as f:
        for item in targetlist:
            seg_list = jieba.cut(item)
            words = []
            for word in seg_list:
                if word not in stopwords_expend:
                    words.append(word)
            words = ' '.join(words)
            f.write(words + '\n')
        f.close()
    
    # 词频统计
    word_lst = []  
    word_dict= {}  
    wf = open('patent_segment.txt')
    wf2 = open("words_count.txt",'w')
    for word in wf:  
        word_lst.append(word.split(' '))  
    for item in word_lst:  
        for item2 in item:  
            if item2 not in word_dict: 
                word_dict[item2] = 1  
            else:  
                word_dict[item2] += 1  
  
    word_list = sorted(word_dict.items(), key=lambda item:item[1], reverse=True)
    word_dict = dict(word_list)
    for key in word_dict:    
        wf2.write(key+' '+str(word_dict[key])+'\n') 
    wf.close()
    wf2.close()
    
    # 显示
#     data_count = pd.Series(word_dict)
#     data_count.plot(kind='bar')

def kmeans(path, target_col, target_file):
    f = open(path).readlines()
    model_tfidf = text.TfidfVectorizer('content')
    title_vector = model_tfidf.fit_transform(f)
    model_cluster = k_means(title_vector, n_clusters=20, random_state=0)
    labels = model_cluster[1]
    data['cluster_label'] = labels
    data_cluster = data[[u'申请号', target_col, 'cluster_label']]
    data_cluster.to_csv(target_file, encoding='utf-8-sig')
    f.close()


def minibatch_kmeans(path, target_col, target_file):
    f = open(path).readlines()
    model_tfidf = text.TfidfVectorizer('content')
    title_vector = model_tfidf.fit_transform(f)
    model_cluster = MiniBatchKMeans(n_clusters=20, random_state=0)
    model_cluster.fit_transform(title_vector)
    labels = model_cluster.labels_
    data['cluster_label'] = labels
    data_cluster = data[[u'申请号', target_col, 'cluster_label']]
    data_cluster.to_csv(target_file, encoding='utf-8-sig')
    f.close()

def spectral(path, target_col, target_file):
    f = open(path).readlines()
    model_tfidf = text.TfidfVectorizer('content')
    title_vector = model_tfidf.fit_transform(f)
    title_vector = kneighbors_graph(title_vector, n_neighbors=10)
    title_vector = title_vector.toarray()

    labels = spectral_clustering(title_vector, n_clusters=100)
    print(len(labels))
    data['cluster_label'] = labels
    data_cluster = data[[u'申请号', target_col, 'cluster_label']]
    data_cluster.to_csv(target_file, encoding='utf-8-sig')
    f.close()

def agnes(path, target_col, target_file):
    f = open(path).readlines()
    model_tfidf = text.TfidfVectorizer('content')
    title_vector = model_tfidf.fit_transform(f)
    title_vector = kneighbors_graph(title_vector, n_neighbors=10)
    title_vector = title_vector.toarray()

    model = AgglomerativeClustering(n_clusters=100)
    model.fit(title_vector)
    labels = model.labels_
    data['cluster_label'] = labels
    data_cluster = data[[u'申请号', target_col, 'cluster_label']]
    data_cluster.to_csv(target_file, encoding='utf-8-sig')
    f.close()


# In[271]:

# 申请日统计
def appli_year(filepath):
    pathlist = view_eachfile(filepath)
    yeardict = {}
    for path in pathlist:
        data_path = pd.read_excel(path, header=0)
        data_path.fillna(0, inplace=True)
        for i in range(len(data_path)):
            year = data_path.ix[i,u'申请日']
            year = remain(year, 4)
            if year not in yeardict:
                yeardict[year] = 1
            else:
                yeardict[year] += 1
    year_list = sorted(yeardict.items(), key=lambda item:item[1], reverse=True)
    yeardict = dict(year_list)
    
    year_count = pd.Series(yeardict)
    year_count.to_csv("applicat_year_count.csv", encoding='utf-8-sig')
    
    
# 来源国统计
def prior_country_count(filepath):
    pathlist = view_eachfile(filepath)
    priordict = {}
    for path in pathlist:
        data_path = pd.read_excel(path, header=0)
        data_path.fillna(0, inplace=True)
        for i in range(len(data_path)):
            prior = data_path.ix[i,u'优先权信息']
            prior = remain(prior, 2, 1)
            if prior not in priordict:
                priordict[prior] = 1
            else:
                priordict[prior] += 1
    prior_list = sorted(priordict.items(), key=lambda item:item[1], reverse=True)
    priordict = dict(prior_list)
    
    prior_count = pd.Series(priordict)
    prior_count.to_csv("prior_country_count.csv", encoding='utf-8-sig')
    
    
# 市场国统计
def target_country_count(filepath):
    pathlist = view_eachfile(filepath)
    target_countrydict = {}
    for path in pathlist:
        data_path = pd.read_excel(path, header=0)
        data_path.fillna(0, inplace=True)
        for i in range(len(data_path)):
            target_country = data_path.ix[i,u'申请号']
            target_country = remain(target_country, 2, 1)
            if target_country not in target_countrydict:
                target_countrydict[target_country] = 1
            else:
                target_countrydict[target_country] += 1
    target_country_list = sorted(target_countrydict.items(), key=lambda item:item[1], reverse=True)
    target_country_dict = dict(target_country_list)
    
    target_country = pd.Series(target_country_dict, index=target_country_dict[0])
    target_country.to_csv("target_country_count.csv", encoding='utf-8-sig')


if __name__ == "__main__":
    path = "D:\\Project code\\file\\"
    filepath = view_eachfile(path)
    data = pd.read_excel(filepath[0], header=0)
    data.fillna(0, inplace=True)
    
    # 标题统计
    data_title = data[u'标题']
    title_list = []
    for i in range(len(data)):
        if data.ix[i, u'标题'] == 0:
            continue
        if zhpattern(data.ix[i, u'标题']):
            title_list.append(data.ix[i, u'标题'])
        else:
            title_list.append(data.ix[i, u'标题（翻译）'])
    words_count(title_list)
#     appli_year(path)
#     prior_country_count(path)
    target_country_count(path)
    
    
    





