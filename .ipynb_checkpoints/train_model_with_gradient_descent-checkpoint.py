import numpy as np
import csv
import matplotlib.pyplot as plt

dataset = []
with open('data.csv', 'r') as file:
    reader = csv.reader(file)
    next(reader)
    for row in reader:
        dataset.append([float(row[0]), float(row[1])])

data_array = np.array(dataset)

X_vector = data_array[:, 0]
Ones_vector = np.ones(X_vector.shape)
X_matrix = np.column_stack((X_vector, Ones_vector))
Y_vector = data_array[:, 1]
Y_vector = Y_vector.reshape(Y_vector.shape[0], 1)


# np.set_printoptions(precision=2, suppress=True)
# plt.scatter(X_vector, Y_vector)
# print(X_matrix.shape)
# print(Y_vector.shape)
# print(X_matrix)
# print(Y_vector)

# theta = vector with 2 values, slope and y_intercept
# we create a and b aleatory to have a starting point for our model
theta = np.random.randn(2, 1)
# theta.shape
# theta

# the model => F = X * theta
# returning the matrix product of X and theta
def model(X, theta)
    return X.dot(theta)

# model(X_matrix, theta)
# plt.scatter(X_vector, Y_vector, c='g')
# plt.plot(X_vector, model(X_matrix, theta), c= 'r')
