# -*- coding:utf-8 -*-
# !user/bin/python
import os
import numpy as np


# 生成文档每个词的topic
def sample_topic(file_data, doc_number, topic_number):
    z_m_n = []
    for i in range(doc_number):
        z = []
        for j in range(len(file_data[i])):
            z.extend([np.argmax(np.random.multinomial(20, [1./topic_number]*topic_number))])
        z_m_n.append(z)
    return z_m_n


# 文档/topic, topic/term数目统计
def num_clc(file_data, dictionary, doc_number, topic_number, z_m_n,
            n_topic_doc, n_doc, n_term_topic, n_topic):
    for m in range(doc_number):
        for n in range(len(file_data[m])):
            for topic in range(topic_number):
                if z_m_n[m][n] == topic:
                    n_topic_doc[m][topic] += 1
                    n_doc[m] += 1
                    value = file_data[m][n]
                    index = dictionary.index(value)
                    n_term_topic[topic][index] += 1
                    n_topic[1] += 1
    return n_topic_doc, n_doc, n_term_topic, n_topic


# Gibbs sampling
def gibbs_sampling(file_data, dictionary, doc_number, topic_number, z_m_n,
                   n_topic_doc, n_doc, n_term_topic, n_topic):
    for i in range(doc_number):
        for j in range(len(file_data[i])):
            # decrement
            dec_k = z_m_n[i][j]
            dec_t = dictionary.index(file_data[i][j])
            n_topic_doc[i][dec_k] -= 1
            n_doc[i] -= 1
            n_term_topic[dec_k][dec_t] -= 1
            n_topic[dec_k] -= 1
            # Gibbs sampling
            new_z = []
            for sample in range(topic_number):
                theta_pre = n_topic_doc[i][sample] + alpha_k[sample]
                phi_pre = float(n_term_topic[sample][dec_t] + beta_t[dec_t])/(sum(n_term_topic[sample])
                                                                              + beta_t[dec_t])
                pre = theta_pre * phi_pre
                new_z.append(pre)
            new_z = [new_z[item]/sum(new_z) for item in range(len(new_z))]
            new_k = np.argmax(np.random.multinomial(20, new_z))
            z_m_n[i][j] = new_k
            # increment
            n_topic_doc[i][new_k] += 1
            n_doc[i] += 1
            n_term_topic[new_k][dec_t] += 1
            n_topic[new_k] += 1
    return n_topic_doc, n_doc, n_term_topic, n_topic

if __name__ == "__main__":
    print
    data = []
    total_data = []
    f = open('test_out.txt')
    for line in f:
        line = line.strip().split(' ')
        total_data.extend(line)
        data.append(line)
    f.close()
    doc_num = len(data)
    topic_num = 5
    dic = list(set(total_data))
    dic_num = len(dic)

    # initialisation
    alpha = 0.5
    alpha_k = [alpha] * topic_num
    beta = 0.5
    beta_t = [beta] * dic_num
    n_k_m = [[0 for t in range(topic_num)] for doc in range(doc_num)]
    n_m = [0 for doc in range(doc_num)]
    n_t_k = [[0 for dic_n in range(dic_num)] for t in range(topic_num)]

    n_k = [0 for t in range(topic_num)]
    z_sample = sample_topic(data, doc_num, topic_num)
    n_k_m, n_m, n_t_k, n_k = num_clc(data, dic, doc_num, topic_num, z_sample, n_k_m, n_m, n_t_k, n_k)

    # Gibbs sampling
    iter_num = 1
    for time in range(iter_num):
        n_k_m, n_m, n_t_k, n_k = gibbs_sampling(data, dic, doc_num, topic_num,
                                                z_sample, n_k_m, n_m, n_t_k, n_k)
        print n_k_m, n_m, n_t_k, n_k

    # 参数估计
    phi_k_t = [[(n_t_k[topic][term] + beta_t[term])/(n_k[topic] + beta_t[term])
                for term in range(dic_num)] for topic in range(topic_num)]
    v_m_k = [[(n_k_m[m][k] + alpha_k[k])/(n_m[m] + alpha_k[k])
              for k in range(topic_num)] for m in range(doc_num)]

    # 输出每个文档的主题及主题最相关的五个关键字
    for doc in range(doc_num):
        topic_list = v_m_k[doc]
        topic_list = np.argsort(topic_list)[::-1]
        key = topic_list[0]
        words_list = np.argsort(phi_k_t[key])[::-1]
        words_index = words_list[:5]
        key_words = [dic[index] for index in words_index]

        print '第%d文档:' % doc
        print '主题为：', topic_list
        print '关键字为：', key_words


































