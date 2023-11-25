import csv

def mean(values):
    return sum(values) / float(len(values))

def covariance(x, mean_x, y, mean_y):
    covar = 0.0
    for i in range(len(x)):
        covar += (x[i] - mean_x) * (y[i] - mean_y)
    return covar

def variance(values, mean):
    return sum((x - mean) ** 2 for x in values)

def coefficients(x, y):
    x_mean, y_mean = mean(x), mean(y)
    b1 = covariance(x, x_mean, y, y_mean) / variance(x, x_mean)
    b0 = y_mean - b1 * x_mean
    return b0, b1

def simple_linear_regression(x, y):
    b0, b1 = coefficients(x, y)
    return b0, b1

def train_and_save_model():
    # Load the dataset
    dataset = []
    with open('data.csv', 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            dataset.append([float(row[0]), float(row[1])])

    # Separate dataset into features (mileage) and target variable (price)
    X = [row[0] for row in dataset]
    y = [row[1] for row in dataset]

    # Train the model
    b0, b1 = simple_linear_regression(X, y)

    # Save the trained model
    with open('linear_regression_model.txt', 'w') as file:
        file.write(f'{b0},{b1}')

if __name__ == "__main__":
    train_and_save_model()