import numpy as np
import matplotlib.pyplot as plt

def load_predictions(file_path):
    data = np.loadtxt(file_path)
    Y_vector = data[:, 0]
    predictions = data[:, 1]
    return Y_vector, predictions


# Coefficient of determination (the closest to 1 the better) - methode de moindre carre
def coef_determination(y, pred):
    mean_y = np.mean(y)
    u = ((y - pred)**2).sum()
    v = ((y - mean_y)**2).sum()
    return 1 - (u / v)

# R-squared represents the proportion of the dependent variable's variance 
# that is explained by the independent variable(s)
def plot_coefficient_of_determination(r_squared):
    labels = ['Explained Variance (R-squared)', 'Unexplained Variance']
    values = [r_squared, 1 - r_squared]
    colors = ['lightcoral', 'lightskyblue']

    plt.bar(labels, values, color=colors)
    plt.ylabel('Proportion')
    plt.title('Coefficient of Determination (R-squared)')
    plt.ylim(0, 1)  # Set y-axis limits to represent proportions
    plt.show()

def main():
   try:
       # Coefficient of determination
       Y_vector, predictions = load_predictions('predictions.txt')
       r_squared = coef_determination(Y_vector, predictions)
       plot_coefficient_of_determination(r_squared)
       rounded_r_squared = round(r_squared, 4)
       print(f"Rounded R-squared: {rounded_r_squared}")
   except ValueError as ve:
       print(ve)

if __name__ == "__main__":
    main()