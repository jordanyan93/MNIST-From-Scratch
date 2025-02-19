# src/neural_network.py
import numpy as np
import pickle
from activations import relu, sigmoid, softmax

def initialize_parameters_deep(layer_dims):
    """
    Initialize parameters for a deep neural network with He initialization
    """
    parameters = {}
    L = len(layer_dims)  # number of layers in the network

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

    return parameters

def linear_activation_forward(A_prev, W, b, activation):
    """
    Implement forward propagation for a single layer (linear + activation)
    """
    Z = np.dot(W, A_prev) + b
    
    if activation == "relu":
        A = relu(Z)
    elif activation == "sigmoid":
        A = sigmoid(Z)
    elif activation == "softmax":
        A = softmax(Z)

    # Cache the necessary values for backpropagation
    cache = (A_prev, W, b, Z)
    
    return A, cache

def L_model_forward(X, parameters):
    """
    Implement forward propagation for the entire network
    """
    caches = []
    A = X
    L = len(parameters) // 2  # number of layers in the neural network

    # Hidden layers
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters["W" + str(l)], parameters["b" + str(l)], activation="relu")
        caches.append(cache)

    # Output layer
    AL, cache = linear_activation_forward(A, parameters["W" + str(L)], parameters["b" + str(L)], activation="softmax")
    caches.append(cache)

    return AL, caches

def compute_cost(AL, Y):
    """
    Compute the cost function (cross-entropy loss)
    """
    m = Y.shape[1]
    cost = -1/m * np.sum(np.multiply(Y, np.log(AL)))
    return np.squeeze(cost)  # To ensure the cost is a scalar

def linear_backward(dZ, A_prev, W, b):
    """
    Compute the gradients for a single layer's linear part (dW, db, dA_prev)
    """
    m = A_prev.shape[1]
    dW = 1/m * np.dot(dZ, A_prev.T)
    db = 1/m * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db

def relu_backward(dA, cache):
    """
    Compute the gradient of the loss with respect to the previous layer using ReLU
    """
    A_prev, W, b, Z = cache
    dZ = np.array(dA, copy=True)  # Convert dA to numpy array
    dZ[Z <= 0] = 0

    # Compute the gradients
    dA_prev, dW, db = linear_backward(dZ, A_prev, W, b)
    return dA_prev, dW, db

def softmax_backward(AL, Y, cache):
    """
    Compute the gradient of the loss with respect to the previous layer using softmax
    """
    A_prev, W, b, Z = cache
    dZ = AL - Y
    dA_prev, dW, db = linear_backward(dZ, A_prev, W, b)
    return dA_prev, dW, db

def L_model_backward(AL, Y, caches):
    """
    Perform backpropagation for the entire network
    """
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)  # Ensure Y and AL have the same shape

    # Backpropagate the softmax output layer
    current_cache = caches[L-1]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = softmax_backward(AL, Y, current_cache)

    # Backpropagate through all previous layers (ReLU for hidden layers)
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = relu_backward(grads["dA" + str(l+1)], current_cache)
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads

def update_parameters(parameters, grads, learning_rate):
    """
    Update the parameters (W and b) using the gradients
    """
    L = len(parameters) // 2  # number of layers in the neural network

    for l in range(L):
        parameters["W" + str(l+1)] -= learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] -= learning_rate * grads["db" + str(l+1)]

    return parameters

def L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=True):
    """
    Train the neural network for a given number of iterations
    """
    costs = []
    parameters = initialize_parameters_deep(layers_dims)

    for i in range(num_iterations):
        # Forward propagation
        AL, caches = L_model_forward(X, parameters)

        # Compute the cost
        cost = compute_cost(AL, Y)

        # Backpropagation
        grads = L_model_backward(AL, Y, caches)

        # Update the parameters
        parameters = update_parameters(parameters, grads, learning_rate)

        if print_cost and i % 100 == 0:
            print(f"Cost after iteration {i}: {cost:.4f}")
        
        if print_cost and i % 10 == 0:
            costs.append(cost)

    return parameters