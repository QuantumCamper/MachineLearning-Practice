import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import accuracy_score


def logistic_regression(x, y, alpha=0.1, iter=10000):
	m,n = x.shape
	x = np.mat(x)
	y = np.mat(y).reshape((m, 1))
	theta = np.mat(np.ones((3,1)))

	for i in range(iter):
		diff = y - sigmoid(x * theta)
		theta = theta + (x.T * diff)

	return theta

def categorical(x, y, no_label=0):
	cate_x = x[y != no_label]
	cate_y = y[y != no_label]
	return cate_x, cate_y

# sigmoid函数
def sigmoid(x):
	return 1.0/(1.0 + np.exp(-x))

# softmax函数
def softmax(x):
	numerator = np.exp(x)
	n_sum = np.sum(numerator, axis=1)
	return numerator/n_sum

# softmax 分类器
def softmax_classfy(x, y, iter=1000, alpha=1.0, class_num=3):
	m, n = x.shape
	theta = np.mat(np.random.rand(n, class_num))

	for i in range(iter):
		diff = y - softmax(x * theta) 
		theta = theta + alpha* (x.T * diff)

	return theta






if __name__ == '__main__':
	data = pd.read_csv('iris.data', header=None)
	data[4] = pd.Categorical(data[4]).codes
	x,y = data.values[:, :4], data.values[:, -1]
	# x = x[:, :2]
	# x = np.concatenate([x, np.ones((len(x), 1))], axis=1)

	#标准化
	scaler = StandardScaler()
	x = scaler.fit_transform(x)
	y_true = y

	# 1.logistic regression 
	# x1, y1 = categorical(x, y, no_label=0)
	# x2, y2 = categorical(x, y, no_label=1)
	# x3, y3 = categorical(x, y, no_label=2)

	# weight1 = logist_regression(x1, y1)
	# weight2 = logist_regression(x2, y2)
	# weight3 = logist_regression(x3, y3)

	# y_hat1 = x1*weight1
	# y_hat1 = y_hat1.reshape(y1.shape)
	
	
	# x1_min, x1_max = x[:, 0].min(), x[:, 0].max()   # 第0列的范围
	# x2_min, x2_max = x[:, 1].min(), x[:, 1].max()   # 第1列的范围
	# weight1 = np.asarray(weight1).reshape((3,))
	# weight2 = np.asarray(weight2).reshape((3,))
	# weight3 = np.asarray(weight3).reshape((3,))

    
	# x_standard = np.arange(x1_min, x1_max)
	# y_x1 = (-weight1[2] - weight1[0]*x_standard)/weight1[1]
	# y_x2 = (-weight2[2] - weight2[0]*x_standard)/weight2[1]
	# y_x3 = (-weight3[2] - weight3[0]*x_standard)/weight3[1]
	# mpl.rcParams['font.sans-serif'] = [u'simHei']
	# mpl.rcParams['axes.unicode_minus'] = False
	# cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])
	# plt.figure(facecolor='w')
	# plt.scatter(x[:, 0], x[:, 1], c=y, edgecolors='k', s=50, cmap=cm_dark)    # 样本的显示
	# plt.plot(x_standard, y_x1, 'r')
	# plt.plot(x_standard, y_x2, 'g')
	# plt.plot(x_standard, y_x3, 'b')
	# plt.xlabel(u'花萼长度', fontsize=14)
	# plt.ylabel(u'花萼宽度', fontsize=14)
	# plt.xlim(x1_min, x1_max)
	# plt.ylim(x2_min, x2_max)
	# plt.grid()
	# plt.show()

	# 2. softmax分类
	y = np.mat(pd.get_dummies(y))
	weights = softmax_classfy(x, y)
	y_hat = np.argmax(x*np.mat(weights), axis=1)
	
	score = accuracy_score(y_true, y_hat)
	print(score)

	


	
	
