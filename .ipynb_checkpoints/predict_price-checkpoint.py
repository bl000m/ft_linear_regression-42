def load_model():
    # Load the trained model
    with open('linear_regression_model.txt', 'r') as file:
        b0, b1 = map(float, file.read().split(','))
    return b0, b1

# calculates the predicted price using the linear regression equation: price = b0 + b1 * mileage.
def predict_price(mileage, b0, b1):
    # Predict the price
    price = b0 + b1 * mileage
    return price

if __name__ == "__main__":
    # Example usage
    b0, b1 = load_model()
    mileage = float(input("Enter the mileage of the car: "))
    estimated_price = predict_price(mileage, b0, b1)
    print(f"Estimated price for {mileage} miles: {estimated_price}")


# def load_model():
#     # Load the trained model
#     with open('linear_regression_model.txt', 'r') as file:
#         final_theta = np.array([[float(file.readline().strip())] for _ in range(2)])
#         mean_X = float(file.readline().strip())
#         std_X = float(file.readline().strip())
#         mean_Y = float(file.readline().strip())
#         std_Y = float(file.readline().strip())
#     return final_theta, mean_X, std_X, mean_Y, std_Y



# def predict_price(km, final_theta, mean_X, std_X, mean_Y, std_Y):
#     # Scale the km value using the same scaling parameters used in training
#     scaled_km = (km - mean_X) / std_X
#     scaled_km_with_bias = np.array(([scaled_km, 1]))
#     # final_theta_reshaped = final_theta.ravel()
#     price = np.dot(scaled_km_with_bias, final_theta)
    
#     # Rescale the predicted price to the original scale
#     estimated_price = price * std_Y + mean_Y
#     return estimated_price


# if __name__ == "__main__":
#     # Example usage
#     final_theta, mean_X, std_X, mean_Y, std_Y = load_model()
#     km = float(input("Enter the km of the car: "))
#     estimated_price = predict_price(km, final_theta, mean_X, std_X, mean_Y, std_Y)
#     print(f"Estimated price for {mileage} miles: {estimated_price}")
