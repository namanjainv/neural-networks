import numpy as np
import itertools

def squared_l2_norm(w):
    """
    Implements the squared L2 norm for weight regularization. ||W||^2
    :param w: column vector of weights [n, 1]
    :return: squared l2 norm of w
    """
    # raise Warning("You must implement squared_l2_norm!")
    return np.sum(np.square(w))

def mean_squared_error(y_hat, y):
    """
    Implements the mean squared error cost function for linear regression
    :param y_hat: predicted values (model output), vector [n, 1]
    :param y: target values vector [n,1]
    :return: mean squared error (scalar)
    """
    # raise Warning("You must implement mean_squared_error!")
    return np.sum(np.square(y_hat - y))/np.max(y.shape)

def calculate_batches(X, batch_size):
    """
    Already implemented, don't worry about it!
    :param X:
    :param batch_size:
    :return:
    """
    indices = list(range(X.shape[0]))
    np.random.shuffle(indices)
    args = [iter(indices)] * batch_size
    batch_indices = itertools.zip_longest(*args, fillvalue=None)
    return [list(j for j in i if j is not None) for i in batch_indices]

class LinearRegression(object):
    def __init__(self, input_dimensions=2, seed=1234):
        """
        Initialize a linear regression model
        :param input_dimensions: The number of features of the input data, for example (height, weight) would be two features.
        """
        np.random.seed(seed)
        self.input_dimensions = input_dimensions
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initialize the weights of a linear regression model, initalize using random numbers.
        """
        # raise Warning("You must implement _initialize_weights! This function should initialize (or re-initialize) your model weights. Use 'the bias trick for this assignment'")
        self.weights = np.random.rand(self.input_dimensions+1,1)

    def fit(self, X_train, y_train, X_val, y_val, num_epochs=10, batch_size=16, alpha=0.01, _lambda=0.0):
        """
        Stochastic Gradient Descent training loop for a linear regression model
        :param X_train: Training input data [n_samples, n_features+1], assume last column is 1's for the bias
        :param y_train: Training target values [n_samples, 1]
        :param X_val: Validation input data [n_samples, n_features+1], assume last column is 1's for the bias
        :param y_val: Validation target values [n_samples, 1]
        :param num_epochs: Number of training epochs (default 10)
        :param batch_size: Number of samples in a single batch (default 16)
        :param alpha: Learning rate for gradient descent update (default 0.01)
        :param _lambda: Coefficient for L2 weight regularization (default 0.0)
        :return: (train_error_, val_error) two arrays of MSE values for each epoch (one for training and validation error)
        """
        # raise Warning("You must implement fit! This function should implement a mini-batch stochastic gradient descent training loop")

        train_error = [] # append your MSE on training set to this after each epoch
        val_error = [] # append your MSE on validation set to this after each epoch

        batch_indices = calculate_batches(X_train, batch_size)
        for epoch in range(num_epochs):
            for batch in batch_indices:
                self._train_on_batch(X_train[batch], y_train[batch], alpha, _lambda)
            train_error.append(mean_squared_error(self.predict(X_train), y_train))
            val_error.append(mean_squared_error(self.predict(X_val), y_val))
            # calculate error on validation set here

        # return: two lists (train_error_after_each epoch, validation_error_after_each_epoch)
        return train_error, val_error

    def predict(self, X):
        """
        Make a prediction on an array of inputs, must already contain bias as last column
        :param X: Array of input [n_samples, n_features+1]
        :return: Array of model outputs [n_samples, 1]
        """
        # raise Warning("You must implement predict. This function should make a prediction on a matrix of inputs")
        return np.dot(X, self.weights)

    def _train_on_batch(self, X, y, alpha, _lambda):
        """
        Given a single batch of data, and the necessary hyperparameters, perform a single batch gradient update. This function should update the model weights.
        :param X: Batch of training input data [batch_size, n_features+1]
        :param y: Batch of training targets [batch_size, 1]
        :param alpha: Learning rate (scalar i.e. 0.01)
        :param _lambda: Regularization strength coefficient (scalar i.e. 0.0001)
        """

        gradients = self._mse_gradient(X, y)
        momentum = np.abs(2*_lambda*gradients)
        self.weights = (self.weights - alpha*(gradients + momentum))
        self.weights = self._l2_regularization_gradient()
        # calculate output
        # calculate errors, mean squared error, and squared L2 regularization
        # calculate gradients of cross entropy and L2  w.r.t weights
        # perform gradient descent update
        # Note: please make use of the functions _mse_gradient and _l2_regularization_gradient
        # raise Warning("You must implement train on batch. This function should perform a stochastic gradient descent update on a single batch of samples")

    def _mse_gradient(self, X, y):
        """
        Compute gradient of MSE objective w.r.t model weights.
        :param X: Set of input data [n_samples, n_features+1]
        :param y: Set of target values [n_samples, 1]
        :return: Gradient of MSE w.r.t model weights [n_features+1, 1]
        """
        # implement the mean squared error gradient for a linear regression model
        # raise Warning("You must implement the gradient. Do not include alpha in your calculation. Gradient should be same dimension as your weights")
        y_hat = self.predict(X)
        return (np.dot((y_hat - y).T, X)/np.max(y.shape)).T

    def _l2_regularization_gradient(self):
        """
        Compute gradient for l2 weight regularization
        :return: Gradient of squared l2 norm w.r.t model weights [n_features+1, 1]
        """
        return self.weights
        #raise Warning("You must implement the gradient for the squared l2 norm of the model weights. Do not include lambda in this part of the calculation")

if __name__ == "__main__":
    print("This is library code. You may implement some functions here to do your own debugging if you want, but they won't be graded")