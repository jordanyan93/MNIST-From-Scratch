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
