import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# 标准线性回归
def lr_metric(xmat, ymat):
	
	xTx = xmat.T * xmat
	if np.linalg.det(xTx) == 0:
		print("can not be inversed")
		return
	else:
		result = np.linalg.inv(xTx) * xmat.T * ymat
		return np.dot(xmat, result)

# 标准线性回归+Ridge
def lr_ridge(xmat, ymat, lamda=0.1):
	
	m, n = xmat.shape
	I = np.eye(n)
	xTx = xmat.T * xmat
	if np.linalg.det(xTx + (lamda*I)) == 0:
		print("can not be inversed")
		return
	else:
		result = np.linalg.inv(xTx + (lamda*I)) * xmat.T * ymat
		return result

# 局部加权线性回归
def lwlr(xmat, ymat, k=1.0):

	m = np.shape(xmat)[0]

	def lwlr_y(target, xmat, ymat, k):
		weight = np.eye(m)
		for i in range(m):
			temp = target - xmat[i]
			mod_value = temp * temp.T
			weight[i, i] = np.exp(np.sqrt(mod_value)/(-2.0*(k**2)))
		xTx = xmat.T * weight * xmat
		if np.linalg.det(xTx) == 0.0:
			print("can not do inversed")
			return
		theta = xTx.I * xmat.T * weight * ymat
		return target * theta
	
	y_hat = np.zeros(m)
	for i in range(m):
		y_hat[i] = lwlr_y(xmat[i], xmat, ymat, k)
	return y_hat

# batch gradient descent
def batch_gd(xmat, ymat, x_test, y_test, iter_show=200, iter=5000, alpha=1e-5):
	m, n = xmat.shape
	iter_error = []
	iter_test_error = []
	theta = np.mat(np.ones((n, 1)))

	for i in range(iter):
		diff = xmat * theta - ymat
		theta = theta - alpha * (xmat.T * diff)
		

		if i % iter_show == 0:
			diff_show = xmat * theta - ymat
			error_train = np.sum(np.abs(diff_show))/m
			print("iter:{}, the training error is: {}".format(i, error_train))
			diff_test_show = x_test * theta - y_test
			error_test = np.sum(np.abs(diff_test_show))/x_test.shape[0]
			print("iter:{}, the test error is: {}".format(i, error_test))

	return theta

# stochastic gradient descent
def stochastic_gd(xmat, ymat, x_test, y_test, iter_show=200, iter=5000, alpha=1e-5):
	m, n = xmat.shape
	iter_error = []
	iter_test_error = []
	theta = np.mat(np.ones((n, 1)))

	for i in range(iter):
		for j in range(m):
			random_index = np.random.randint(0, m)
			target = xmat[random_index].reshape((n, 1))
			diff = xmat[random_index] * theta - ymat[random_index]
			theta = theta - alpha * (target * diff)

		if i % iter_show == 0:
			diff_show = xmat * theta - ymat
			error_train = np.sum(np.abs(diff_show))/m
			print("iter:{}, the training error is: {}".format(i, error_train))
			diff_test_show = x_test * theta - y_test
			error_test = np.sum(np.abs(diff_test_show))/x_test.shape[0]
			print("iter:{}, the test error is: {}".format(i, error_test))
		

	return theta

# stochastic gradient descent + Ridge
def stochastic_gd(xmat, ymat, x_test, y_test, iter_show=200, iter=5000, alpha=1e-5, lamda=1e-4):
	m, n = xmat.shape
	iter_error = []
	iter_test_error = []
	theta = np.mat(np.ones((n, 1)))

	for i in range(iter):
		for j in range(m):
			random_index = np.random.randint(0, m)
			target = xmat[random_index].reshape((n, 1))
			diff = xmat[random_index] * theta - ymat[random_index]
			theta = theta - alpha * (target * diff) - lamda * theta

		if i % iter_show == 0:
			diff_show = xmat * theta - ymat
			error_train = np.sum(np.abs(diff_show))/m
			print("iter:{}, the training error is: {}".format(i, error_train))
			diff_test_show = x_test * theta - y_test
			error_test = np.sum(np.abs(diff_test_show))/x_test.shape[0]
			print("iter:{}, the test error is: {}".format(i, error_test))
		

	return theta

# stochastic gradient descent + Lasso
def stochastic_lasso(xmat, ymat, x_test, y_test, iter_show=200, iter=5000, alpha=1e-5, lamda=1e-4):
	m, n = xmat.shape
	iter_error = []
	iter_test_error = []
	theta = np.mat(np.ones((n, 1)))

	for i in range(iter):
		for j in range(m):
			random_index = np.random.randint(0, m)
			target = xmat[random_index].reshape((n, 1))
			diff = xmat[random_index] * theta - ymat[random_index]
			l1_norm = np.mat(lasso_derivative(theta))
			theta = theta - alpha * (target * diff) - lamda * l1_norm

		if i % iter_show == 0:
			diff_show = xmat * theta - ymat
			error_train = np.sum(np.abs(diff_show))/m
			print("iter:{}, the training error is: {}".format(i, error_train))
			diff_test_show = x_test * theta - y_test
			error_test = np.sum(np.abs(diff_test_show))/x_test.shape[0]
			print("iter:{}, the test error is: {}".format(i, error_test))
		

	return theta

# derivative of L1-norm
def lasso_derivative(w, alpha=10e6):
    w_lasso = []
    for i in range(len(w.flatten())):
        w_la = 1.0/(1 + np.exp(- (alpha * w[i]))) - 1.0/(1 + np.exp(alpha * w[i]))
        w_lasso.append(w_la)
    return np.array(w_lasso)

# minibatch gradient descent
def mini_stochastic_gd(xmat, ymat, x_test, y_test, iter_show=200, iter=5000, alpha=1e-5, mini=100):
	m, n = xmat.shape
	iter_error = []
	iter_test_error = []
	theta = np.mat(np.ones((n, 1)))

	for i in range(iter):
		for j in range(mini):
			random_index = np.random.randint(m, size=mini)
			target = xmat[random_index]
			diff = xmat[random_index] * theta - ymat[random_index]
			theta = theta - alpha * (target.T * diff)

		if i % iter_show == 0:
			diff_show = xmat * theta - ymat
			error_train = np.sum(np.abs(diff_show))/m
			print("iter:{}, the training error is: {}".format(i, error_train))
			diff_test_show = x_test * theta - y_test
			error_test = np.sum(np.abs(diff_test_show))/x_test.shape[0]
			print("iter:{}, the test error is: {}".format(i, error_test))
		

	return theta




if __name__ == '__main__':
	data = pd.read_table('housing.data', header=None, sep='\s+')
	x = data.iloc[:,:13]
	y = data.iloc[:, -1]
	x = np.mat(x).reshape((len(x), 13))
	y = np.mat(y).reshape((len(y), 1))

	scaler = StandardScaler()
	scaler.fit(x)
	x = scaler.transform(x)
	poly = PolynomialFeatures(2)
	x = poly.fit_transform(x)

	x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=0)
	x_train, x_test, y_train, y_test = np.mat(x_train), np.mat(x_test), np.mat(y_train), np.mat(y_test) 
	
	weights = stochastic_lasso(x_train, y_train, x_test, y_test)
	
	y_hat = x_test * weights
	R2 = r2_score(y_test, y_hat)
	mse = mean_squared_error(y_test, y_hat)

	print("R2 score is: ", R2)
	print("mean squared error is: ", mse)

	# 可视化
	# 1. 测试集真实值和预测值
	mpl.rcParams['font.sans-serif'] = [u'simHei']
	mpl.rcParams['axes.unicode_minus'] = False
	x_axis = np.arange(x_test.shape[0])
	plt.figure(figsize=(8,10), facecolor='w')
	plt.plot(x_axis, y_test, 'r', linewidth=1, label=u'真实值')
	plt.plot(x_axis, y_hat, 'g', linewidth=1, label=u'预测值')
	plt.legend(loc='best')
	plt.title(u'波士顿房价预测')
	plt.xlabel(u'样本数')
	plt.ylabel(u'房屋价格')
	plt.show()
	



	

	
	
	# corrcoef = np.corrcoef(y_hat, y.flatten().A[0])
	
