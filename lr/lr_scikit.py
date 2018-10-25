from sklearn import datasets, linear_model
import numpy as np

# height (cm) - matrix X
X = np.array([[147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183]]).T

# weight (kg) - matrix y
y = np.array([[49, 50, 51, 54, 58, 59, 60, 62, 63, 64, 66, 67, 68]]).T

# Calculate Xbar - insert one row 1 before X
# Get column 1 and its number of elements, create a matrix of column 1's size
ones = np.ones((X.shape[0], 1))
# Now we create Xbar = [col1, col2] = [ones, X] => y = X * w1 + ones * w = W * Xbar
Xbar = np.concatenate((ones, X), axis=1)

# fit the model by Linear Regression
regr = linear_model.LinearRegression(fit_intercept=False)
regr.fit(Xbar, y)

print ('Solution w: ', regr.coef_)
