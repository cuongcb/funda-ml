from __future__ import print_function, unicode_literals, division
from sklearn import datasets, linear_model
import numpy as np

# init X, y
X = []
y = []

with open('ex1data2.txt') as f:
	for line in f:
		line = line.replace('\n','')
		vals = line.split(',')
		X.append(vals[:2])
		y.append(vals[2])

X = np.array(X)
y = np.array(y)

# size of data set
m = X.shape[0]

# create Xbar by concatenate ones with X
# Xbar = [1, X] => Xbar * w = 1 * w0 + X1 * w1 + X2 * w2 = y
Xbar = np.concatenate((np.ones((m, 1)), X), 1)
y = np.reshape(y, (m, 1))

# fit the model by Linear Regression
regr = linear_model.LinearRegression(fit_intercept=False)
regr.fit(Xbar, y)

print('Solution theta as scikit-learn: ', regr.coef_)
