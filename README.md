# MNIST Digit Recognition from Scratch using NumPy

This project implements a neural network to classify handwritten digits from the MNIST dataset using only **NumPy**. The model is built from scratch, without relying on high-level libraries like TensorFlow or PyTorch, to help understand the foundational concepts of machine learning and deep learning.

## Problem Statement

The task is to recognize handwritten digits (0-9) from the **MNIST** dataset, which consists of 60,000 training and 10,000 testing grayscale images of digits, each of size 28x28 pixels.

## Key Features

- **Implemented a shallow neural network**: The network consists of multiple layers with ReLU activations for hidden layers and softmax for the output layer.
- **Backpropagation from scratch**: Manually implemented forward and backward propagation using basic matrix operations in NumPy.
- **Gradient Descent**: Used to optimize the model weights based on the computed gradients.
- **Accuracy Measurement**: Evaluated the model on the MNIST test dataset to compute the accuracy and visualize predictions.

## Technologies Used

- **Python** (Programming Language)
- **NumPy** (Library for array manipulations and mathematical operations)
- **Matplotlib** (For visualizing the results)

## Installation

To install dependencies:

```bash
pip install -r requirements.txt
```
## Usage
Once you have installed the required dependencies (by running pip install -r requirements.txt), you can run the main script with the following command:

```bash
python src/main.py
```
This will start the program, and you will be prompted with a menu where you can:
1. Test predictions on specific image indices.
2. Retrain the model.
3. Exit the program.
   
Make sure that the train.csv and test.csv files are present in the /data directory before running the script.

## Directory Structure
```
/data                     # Contains the MNIST dataset files
    /train.csv            # Training data
    /test.csv             # Testing data
/src                      # Source code for the project
    /main.py              # Main script for running the model
    /neural_network.py    # Contains the neural network functions
    /activations.py       # Contains activation function implementations
    /data_preprocessing.py# Contains data preprocessing functions
/notebooks/               # Jupyter notebooks for experimentation (if applicable)
/model.pkl                # Saved model parameters (pickled file)
/requirements.txt         # Required dependencies
README.md                 # Project overview and setup instructions
.gitignore
```

## Retraining the Model
If you'd like to retrain the model, you can select the "Retrain the model" option in the menu. This will retrain the model on the MNIST dataset and save the updated model parameters to model.pkl.
