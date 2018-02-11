import pandas as pd
import numpy as np


# 计算熵
def entropy(x, m):
	temp = float(x/m)
	return -temp*np.log2(temp)

# 计算样本经验熵
def emp_entropy(data):
	class_nums = data.groupby(data.iloc[:, -1]).size().values
	m ,n = data.shape
	H_D = np.sum(entropy(x, m) for x in class_nums)
	return H_D

# 计算经验条件熵
def entropy_condition(data, n_split):
	total_len = len(data)
	num_class = set(data.iloc[:, n_split])
	condition_entropy = 0
	for part in num_class:
		sub_data = data[data.iloc[:, n_split] == part]
		sub_classes = sub_data.groupby(sub_data.iloc[:, -1]).size().values
		sub_len = len(sub_data)
		sub_sum = np.sum(entropy(x, sub_len) for x in sub_classes)
		condition_entropy += (sub_len / total_len) * sub_sum

	return condition_entropy

# 计算基尼系数
def gini(data):
	classes = data.groupby([data.iloc[:,-1]]).size().values
	m, n = data.shape
	gini = 1.0 - np.sum((x/m)**2 for x in classes)
	return gini
				
# 计算每一列最小基尼系数分割点
def gini_col(data):
	m, n = data.shape
	col_split = {}
	for col in range(n-1):
		point_split = list(set(data.iloc[:, col]))
		gini_list = []
		for num in point_split:
			data_left = data[data.iloc[:, col] <= num]
			data_right = data[data.iloc[:, col] > num]
			data_gini = data_left.shape[0]/m * gini(data_left) + data_right.shape[0]/m * gini(data_right)
			gini_list.append(data_gini)

		min_gini = min(gini_list)
		index_min = gini_list.index(min_gini)
		col_split[col] = [point_split[index_min], min_gini]

	min_split_col = min(col_split.items(), key=lambda x: x[1][1])
	return min_split_col[0], min_split_col[1][0], min_split_col[1][1]

# CART 算法
def cart_tree(data, labels='labels'):
	# 如果结果都是同一类，停止
	if data.groupby(labels).size()[0] == len(data):
		return data[labels].values[0]
	# 如果只剩一列，停止:
	if len(data.columns) == 1:
		return data.groupby(labels).size().sort_values().index[-1]

	# 选取gini系数最小的特征
	tree_labels = data.columns.values
	best_split_col, best_split_point, best_split_gini = gini_col(data)
	if best_split_gini < 0.001:
		return data.groupby(labels).size().sort_values().index[-1]

	if len(data.iloc[:, best_split_col]) == 1:
		return data[labels].values[0]

	best_tree_labels = tree_labels[best_split_col]
	key = best_tree_labels + ':' + str(best_split_point)
	decision_tree = {key: {}}

	data_left = data[data[best_tree_labels] <= best_split_point]
	decision_tree[key][0] = cart_tree(data_left, labels='labels')
	data_right = data[data[best_tree_labels] > best_split_point]
	decision_tree[key][1] = cart_tree(data_right, labels='labels')

	

	return decision_tree


# 选取最大增益
def choose_best_feature(data, method='ID3'):
	m, n = data.shape
	
	if method == 'ID3':
		entropy_split = [entropy_condition(data, n_split) for n_split in range(n-1)]
		return entropy_split.index(max(entropy_split))
	elif method == 'C4.5':
		entropy_split = [np.sum(entropy(item, m) for item in data.groupby(data.iloc[:, n_split]).size().values) for 
		        n_split in range(n-1)]
		return entropy_split.index(max(entropy_split))

# ID3/C4.5算法
def create_tree(data, labels='labels', method='ID3'): 

	# 如果结果都是同一类，停止
	if data.groupby(labels).size()[0] == len(data):
		return data[labels].values[0]
	# 如果只剩一列，停止:
	if len(data.columns) == 1:
		return data.groupby(labels).size().sort_values().index[-1]

	# 选取信息增益最大的特征
	tree_labels = data.columns.values
	best_split = choose_best_feature(data, method)
	best_tree_labels = tree_labels[best_split]
	decision_tree = {best_tree_labels: {}}
	features = set(data[best_tree_labels])
	

	for item in features:
		sub_data = data[data[best_tree_labels]==item]
		
		sub_data.drop([best_tree_labels], axis=1, inplace=True)
		decision_tree[best_tree_labels][item] = create_tree_id3(sub_data, labels='labels')

	return decision_tree

	


if __name__ == '__main__':
	data = pd.read_csv('iris.data', header=None)
	data.columns = ['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width', 'labels']
	class_names = ['Iris Setosa', 'Iris Versicolour', 'Iris Virginica']
	# H_D = emp_entropy(data, labels='labels')
	test = entropy_condition(data, 1)
	
	# print(create_tree(data, method='C4.5'))
	print(cart_tree(data))

