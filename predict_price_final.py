import numpy as np

def load_model(file_path='linear_regression_model.txt'):
    # Load the trained model
    with open(file_path, 'r') as file:
        final_theta = np.array([[float(file.readline().strip())] for _ in range(2)])
        mean_X = float(file.readline().strip())
        std_X = float(file.readline().strip())
        mean_Y = float(file.readline().strip())
        std_Y = float(file.readline().strip())
    return final_theta, mean_X, std_X, mean_Y, std_Y

def predict_price(km, final_theta, mean_X, std_X, mean_Y, std_Y):
    # Validate input
    if km < 0:
        raise ValueError("Hey, a car with less than 0 km doesn't exist!")
    elif km > 400000:
        raise ValueError("Whoa, with this many kilometers, our crystal ball is a bit cloudy")

    # Scale the km value using the same scaling parameters used in training
    scaled_km = (km - mean_X) / std_X
    scaled_km_with_bias = np.array([[scaled_km, 1]])
    price = np.dot(scaled_km_with_bias, final_theta)
    
    # Rescale the predicted price to the original scale, extract scalar value, and round it
    estimated_price = round(np.squeeze(price) * std_Y + mean_Y)
    return estimated_price

def main():
    try:
        # Example usage
        final_theta, mean_X, std_X, mean_Y, std_Y = load_model()
        km = float(input("Enter the km of the car: "))
        estimated_price = predict_price(km, final_theta, mean_X, std_X, mean_Y, std_Y)
        print(f"Estimated price for {km} km: {estimated_price}")
    except ValueError as ve:
        print(ve)

if __name__ == "__main__":
    main()
