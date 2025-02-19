import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_preprocessing import *  # Import your data preprocessing functions
from data_preprocessing import load_data
from neural_network import *      # Import neural network functions
from activations import *         # Import activation functions
import pickle  # For saving and loading the model


# Make predictions based on the trained parameters
def make_predictions(parameters, X):
    """
    Given a dataset X and trained parameters, predict the class labels.
    """
    probabilities, caches = L_model_forward(X, parameters)
    predictions = np.argmax(probabilities, axis=0)
    return predictions


# Test prediction for a specific index
def test_prediction(index, parameters, X_test):
    """
    Given an index, display the image at that index, predict its label, and print the prediction.
    
    Arguments:
    - index : int : index of the image to be tested.
    - parameters : dict : Trained neural network parameters.
    - X_test : ndarray : Test dataset (images).
    """
    # Extract the current image
    current_image = X_test[:, index, None]

    # Predict the class for the current image
    prediction = make_predictions(parameters, current_image)
    prediction_label = np.squeeze(prediction)  # Get rid of redundant dimensions
    
    # Print the prediction to the terminal
    print(f"Prediction for index {index}: {prediction_label}")
    
    # Reshape and scale the image for proper display
    image_to_show = current_image.reshape((28, 28)) * 255
    
    # Display the image with prediction as title
    plt.gray()
    plt.imshow(image_to_show, interpolation='nearest')
    plt.title(f"Predicted Label: {prediction_label}")  # Add prediction label as title
    plt.show()


# Train the model
def train_model(X_train, Y_train):
    """
    Train the neural network using the training data.
    """
    # Define the architecture of the neural network
    layers_dims = [784, 10, 6, 10]  # Adjust the layer sizes as needed
    
    # Train the model and get the parameters
    parameters = L_layer_model(X_train, Y_train, layers_dims, learning_rate=0.0075, num_iterations=1000, print_cost=True)
    
    return parameters


# Save model parameters to a file
def save_model(parameters, filename='model.pkl'):
    """
    Save the trained model parameters to a file using pickle.
    """
    with open(filename, 'wb') as f:
        pickle.dump(parameters, f)
    print(f"Model saved to {filename}")


# Load model parameters from a file
def load_model(filename='model.pkl'):
    """
    Load model parameters from a file using pickle.
    """
    with open(filename, 'rb') as f:
        parameters = pickle.load(f)
    print(f"Model loaded from {filename}")
    return parameters


# Interactive loop to interact with the user
def interactive_loop(X_train, Y_train, X_test):
    # Check if the model already exists
    model_filename = 'model.pkl'
    if os.path.exists(model_filename):
        # If model exists, load the model parameters
        parameters = load_model(model_filename)
        print("Model loaded successfully.")
    else:
        # If no saved model, train a new model
        parameters = train_model(X_train, Y_train)
        
        # Save the trained model
        save_model(parameters, model_filename)
        print("Model trained and saved.")

    # Main loop for user interaction
    while True:
        # Ask the user for input
        user_input = input("Enter an option:\n"
                           "1. Test a specific image index\n"
                           "2. Retrain the model\n"
                           "3. Exit\n"
                           "Your choice: ")

        if user_input == "1":
            # Ask for index and test prediction
            index = int(input("Enter the index of the image to test: "))
            test_prediction(index, parameters, X_test)

        elif user_input == "2":
            # Retrain the model
            print("Retraining the model...")
            parameters = train_model(X_train, Y_train)
            save_model(parameters)  # Save the newly trained model
            print("Model retrained and saved.")

        elif user_input == "3":
            # Exit the loop
            print("Exiting program.")
            break

        else:
            # Handle invalid input
            print("Invalid choice. Please choose again.")


# Main execution
if __name__ == "__main__":
    # Step 1: Load the raw data
    numbers_df, test_df = load_data()
    
    # Step 2: Preprocess the data
    sample_size = 10000  # Adjust sample size if needed
    X_train, Y_train, X_dev, Y_dev, X_test = preprocess_data(numbers_df, test_df, sample_size)

    # Run the interactive loop
    interactive_loop(X_train, Y_train, X_test)
