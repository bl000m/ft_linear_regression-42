import numpy as np

def mse(y_true, y_pred):
   return np.mean((y_true - y_pred) ** 2)

def gradient_descent(X, y, b0, b1, learning_rate, n_iters):
   for _ in range(n_iters):
       y_pred = b0 + b1 * X
       dJ_db0 = -2/len(X) * np.sum(y - y_pred)
       dJ_db1 = -2/len(X) * np.sum(X * (y - y_pred))
       b0 = b0 - learning_rate * dJ_db0
       b1 = b1 - learning_rate * dJ_db1
   return b0, b1

# Load the dataset
dataset = []
with open('data.csv', 'r') as file:
   reader = csv.reader(file)
   next(reader) # Skip header
   for row in reader:
       dataset.append([float(row[0]), float(row[1])])

# Separate dataset into features (mileage) and target variable (price)
X = np.array([row[0] for row in dataset])
y = np.array([row[1] for row in dataset])

# Initialize parameters
b0, b1 = 0, 0

# Perform gradient descent
b0, b1 = gradient_descent(X, y, b0, b1, learning_rate=0.01, n_iters=100)

# Calculate the MSE (minimized)
mse_value = mse(y, b0 + b1 * X)


print(f"MSE: {mse_value}")