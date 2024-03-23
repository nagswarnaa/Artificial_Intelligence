# Swarna, Nagaraju
# 1002_031_714
# 2023_02_26
# Assignment_01_01

import numpy as np


def init_weights(X_train, layers, seed):
    columns_of_first_layer = X_train.shape[0]
    rows_of_first_layer = layers[0]
    np.random.seed(seed)
    first_layer = np.random.randn(
        rows_of_first_layer, columns_of_first_layer+1)
    list_of_weights = []
    list_of_weights.append(first_layer)
    i = 1
    while i < len(layers):
        np.random.seed(seed)
        rows = layers[i]
        columns = layers[i-1]+1
        x = np.random.randn(rows, columns)
        list_of_weights.append(x)
        i = i+1
    return list_of_weights


def calc_sigmoid(x):
    return 1 / (1 + np.exp(-x))


def get_gradient(term1, term2, h):
    return (term1-term2)/(2*h)


def filter_weights(list_of_weights, updated_weight_matrix, i):
    new_list_of_weights = []
    x = 0
    while x < len(list_of_weights):
        if i != x:
            new_list_of_weights.append(list_of_weights[x])
        else:
            new_list_of_weights.append(updated_weight_matrix)
        x = x+1
    return new_list_of_weights


def get_modified_weights(original_weight, h, row, column):
    h_added = np.copy(original_weight)
    h_deducted = np.copy(original_weight)
    h_added[row, column] = (h_added[row, column] + h)
    h_deducted[row, column] = (h_deducted[row, column] - h)
    return h_added, h_deducted


def init_gradients_array(list_of_weights):
    gradients = []
    for i in range(len(list_of_weights)):
        gradients.append(np.ones(list_of_weights[i].shape))
    return gradients


def find_gradient(X_train, Y_train, list_of_weights, h):
    list_of_gradients = init_gradients_array(list_of_weights)
    for i in range(len(list_of_weights)):
        for j in range(list_of_weights[i].shape[0]):
            for k in range(list_of_weights[i].shape[1]):
                h_added, h_deducted = get_modified_weights(
                    list_of_weights[i], h, j, k)
                modified_weights = filter_weights(list_of_weights, h_added, i)
                h_added_output = forward_pass(modified_weights, X_train)
                modified_weights = filter_weights(
                    list_of_weights, h_deducted, i)
                h_deducted_output = forward_pass(modified_weights, X_train)
                list_of_gradients[i][j, k] = get_gradient(
                    calculate_error(Y_train, h_added_output), calculate_error(Y_train, h_deducted_output), h)
    return list_of_gradients


def adjust_weights(X_train, Y_train, list_of_weights, h, alpha):
    list_of_gradients = find_gradient(X_train, Y_train, list_of_weights, h)
    for i in range(len(list_of_weights)):
        list_of_weights[i] = (list_of_weights[i] -
                              (alpha * list_of_gradients[i]))
    return list_of_weights


def calculate_error(original_outcome, predicted_outcome):
    sqaured = np.square(predicted_outcome - original_outcome)
    return np.mean(sqaured)


def forward_pass(list_of_weights, input):
    shape_of_bias = input.shape[1]
    for i in list_of_weights:
        bias_column = np.ones((1, shape_of_bias))
        biased_weights = np.vstack((bias_column, input))
        dot_product = np.dot(i, biased_weights)
        Y = dot_product
        input = calc_sigmoid(Y)
    return input


def train_network(list_of_weights, X_train, Y_train, X_test, Y_test, h, alpha, epochs):
    list_of_errors = []
    for i in range(epochs):
        initial_output = forward_pass(list_of_weights, X_test)
        error_initial = calculate_error(Y_test, initial_output)
        list_of_weights = adjust_weights(
            X_train, Y_train, list_of_weights, h, alpha)
        epoch_output = forward_pass(list_of_weights, X_test)
        epoch_error = calculate_error(Y_test, epoch_output)
        list_of_errors.append(epoch_error)
    final_output = forward_pass(list_of_weights, X_test)
    return [list_of_weights, list_of_errors, final_output]


def multi_layer_nn(X_train, Y_train, X_test, Y_test, layers, alpha, epochs, h=0.00001, seed=2):

    # This function creates and trains a multi-layer neural Network
    # X_train: Array of input for training [input_dimensions,nof_train_samples]

    # Y_train: Array of desired outputs for training samples [output_dimensions,nof_train_samples]
    # X_test: Array of input for testing [input_dimensions,nof_test_samples]
    # Y_test: Array of desired outputs for test samples [output_dimensions,nof_test_samples]
    # layers: array of integers representing number of nodes in each layer
    # alpha: learning rate
    # epochs: number of epochs for training.
    # h: step size
    # seed: random number generator seed for initializing the weights.
    # return: This function should return a list containing 3 elements:
    # The first element of the return list should be a list of weight matrices.
    # Each element of the list corresponds to the weight matrix of the corresponding layer.

    # The second element should be a one dimensional array of numbers
    # representing the average mse error after each epoch. Each error should
    # be calculated by using the X_test array while the network is frozen.
    # This means that the weights should not be adjusted while calculating the error.

    # The third element should be a two-dimensional array [output_dimensions,nof_test_samples]
    # representing the actual output of network when X_test is used as input.

    # Notes:
    # DO NOT use any other package other than numpy
    # Bias should be included in the weight matrix in the first column.
    # Assume that the activation functions for all the layers are sigmoid.
    # Use MSE to calculate error.
    # Use gradient descent for adjusting the weights.
    # use centered difference approximation to calculate partial derivatives.
    # (f(x + h)-f(x - h))/2*h
    # Reseed the random number generator when initializing weights for each layer.
    # i.e., Initialize the weights for each layer by:
    # np.random.seed(seed)
    # np.random.randn()
    list_of_weights = init_weights(X_train, layers, seed)
    return train_network(list_of_weights, X_train, Y_train,
                         X_test, Y_test, h, alpha, epochs)
