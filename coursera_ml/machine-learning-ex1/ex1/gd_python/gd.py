from __future__ import print_function, unicode_literals, division
import numpy as np
# import matplotlib.pyplot as plt

# init X, y
X = []
y = []

with open('ex1data2.txt') as f:
	for line in f:
		line = line.replace('\n','')
		vals = line.split(',')
		X.append(vals[:2])
		y.append(vals[2])

# visualize
# plt.plot(X, y, 'ro')

# must be specific `dtype`, if not => X members are strings.
X = np.array(X, dtype=int)
y = np.array(y, dtype=int)

# size of data set
m = X.shape[0]

# create Xbar by concatenate ones with X
# Xbar = [1, X] => Xbar * w = 1 * w0 + X1 * w1 + X2 * w2 = y
Xbar = np.concatenate((np.ones((m, 1)), X), 1)
y = np.reshape(y, (m, 1))

print(Xbar)
# print(Xbar.T)

# w = pinv(X' * X) * X' * y
A = np.dot(Xbar.T, Xbar)
b = np.dot(Xbar.T, y)
theta = np.dot(np.linalg.pinv(A), b)

print('Solution theta as normal equation: ', theta)
