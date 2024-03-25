import numpy as np
import tensorflow as tf


def init_weights(X_train, layers, seed):
    columns_of_first_layer = X_train.shape[1]+1
    rows_of_first_layer = layers[0]
    np.random.seed(seed)
    list_of_weights = []
    first_layer = np.random.randn(columns_of_first_layer,
                                  rows_of_first_layer)
    list_of_weights.append(first_layer)
    i = 1
    while i < len(layers):
        np.random.seed(seed)
        rows = layers[i-1]+1
        columns = layers[i]
        x = np.random.randn(rows, columns)
        list_of_weights.append(x)
        i = i+1
    return list_of_weights


def init_error_list(epochs):
    errors = []
    for i in range(epochs):
        errors.append(0)
    return errors


def init_output_array(rows, columns):
    return tf.Variable(np.random.randn(rows,
                                       columns))


def split_data(X_train, Y_train, split_range):
    start = int(split_range[0] * X_train.shape[0])
    end = int(split_range[1] * X_train.shape[0])
    return np.concatenate((X_train[:start], X_train[end:])), np.concatenate(
        (Y_train[:start], Y_train[end:])), X_train[start:end], Y_train[start:end]


def generate_batches(X, y, batch_size):
    for i in range(0, X.shape[0], batch_size):
        yield X[i:i+batch_size], y[i:i+batch_size]
    if X.shape[0] % batch_size != 0:
        yield X[-(X.shape[0] % batch_size):], y[-(X.shape[0] % batch_size):]


def activation_functions(X, act_string):
    low_act_string = act_string.lower()
    if low_act_string == "linear":
        return X
    elif low_act_string == "sigmoid":
        return tf.nn.sigmoid(X)
    elif low_act_string == "relu":
        return tf.nn.relu(X)


def calc_forward_pass(list_of_weights, input, activations):
    shape_of_bias = input.shape[0]
    for i, act in zip(list_of_weights, activations):
        biased_weights = tf.concat(
            [tf.ones((shape_of_bias, 1), dtype=tf.float64), input], axis=1)
        dot_product = tf.matmul(biased_weights, i)
        input = activation_functions(dot_product, act)
    return input


def calculate_error(original_outcome, predicted_outcome, error_str):
    error_string = error_str.lower()
    if error_string == "mse":
        squared = tf.square(predicted_outcome - original_outcome)
        return tf.reduce_mean(squared)
    elif error_string == "cross_entropy":
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=original_outcome, logits=predicted_outcome))
    elif error_string == "svm":
        factor = original_outcome * predicted_outcome
        hinge_loss = tf.nn.relu(1. - factor)
        mean_hinge_loss = tf.reduce_mean(hinge_loss)
        return mean_hinge_loss


def toTensors(weights):
    wts = []
    for x in weights:
        wts.append(np.array(x, dtype=np.float32))
    weights = [np.array(w, dtype=np.float32) for w in weights]
    return [tf.Variable(w, dtype=tf.float64) for w in weights]


def find_gradients(weights, X_batch, activations, y_batch, loss, alpha):
    with tf.GradientTape() as tape:
        out_temp = calc_forward_pass(weights, X_batch, activations)
        loss_temp = calculate_error(y_batch, out_temp, loss)
    grads = tape.gradient(loss_temp, weights)
    for i, grad in enumerate(grads):
        weights[i].assign_sub(alpha * grad)
    return weights


def buildNetwork(X_train, Y_train, layers, activations, alpha, batch_size, epochs, loss,
                 validation_split, weights, seed):
    train_X, train_Y, validation_X, validation_Y = split_data(
        X_train, Y_train, validation_split)
    if (weights == None):
        weights = init_weights(X_train, layers, seed)
    error_list = init_error_list(epochs)
    output_list = init_output_array(len(validation_X), Y_train.shape[1])
    weights = toTensors(weights)
    for epoch in range(epochs):
        for X_batch, y_batch in generate_batches(train_X, train_Y, batch_size):
            weights = find_gradients(
                weights, X_batch, activations, y_batch, loss, alpha)
        epoch_output = calc_forward_pass(weights, validation_X, activations)
        epoch_error = calculate_error(validation_Y, epoch_output, loss)
        error_list[epoch] = epoch_error
    final_output = calc_forward_pass(weights, validation_X, activations)
    for i in range(len(weights)):
        weights[i] = weights[i].numpy().astype(np.float32)
    return [weights, error_list, final_output]


def multi_layer_nn_tensorflow(X_train, Y_train, layers, activations, alpha, batch_size, epochs=1, loss="svm",
                              validation_split=[0.8, 1.0], weights=None, seed=2):

    return buildNetwork(X_train, Y_train, layers, activations, alpha, batch_size, epochs, loss,
                        validation_split, weights, seed)
