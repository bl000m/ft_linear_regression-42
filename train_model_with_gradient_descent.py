import csv

def mean(values):
    return sum(values) / float(len(values))

def partial_derivative_b0(x, y, b0, b1):
    n = len(x)
    return -2 * sum(y[i] - (b0 + b1 * x[i]) for i in range(n)) / n

def partial_derivative_b1(x, y, b0, b1):
    n = len(x)
    return -2 * sum(x[i] * (y[i] - (b0 + b1 * x[i])) for i in range(n)) / n

def gradient_descent(x, y, learning_rate, epochs):
    # Initialize coefficients
    b0, b1 = 0.0, 0.0

    # Perform gradient descent
    for _ in range(epochs):
        gradient_b0 = partial_derivative_b0(x, y, b0, b1)
        gradient_b1 = partial_derivative_b1(x, y, b0, b1)

        # Update coefficients using gradient descent
        b0 = b0 - learning_rate * gradient_b0
        b1 = b1 - learning_rate * gradient_b1

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

    # Train the model using gradient descent
    learning_rate = 0.01
    epochs = 1000
    b0, b1 = gradient_descent(X, y, learning_rate, epochs)

    # Save the trained model
    with open('linear_regression_model.txt', 'w') as file:
        file.write(f'{b0},{b1}')

if __name__ == "__main__":
    train_and_save_model()
