def load_model():
    # Load the trained model
    with open('linear_regression_model.txt', 'r') as file:
        b0, b1 = map(float, file.read().split(','))
    return b0, b1

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