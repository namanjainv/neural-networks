import numpy as np
import itertools

def squared_l2_norm(w):
    """
    Implements the squared L2 norm for weight regularization. ||W||^2
    :param w: column vector of weights [n, 1]
    :return: squared l2 norm of w
    """
    # raise Warning("You must implement squared_l2_norm! This is for calculating regularization")
    return np.sum(np.square(w))

def binary_cross_entropy(y_hat, y):
    """
    Implements the binary cross-entropy loss function for logistic regression
    :param y_hat: predicted values (model output), vector [n, 1]
    :param y: target values vector [n,1], binary values either 0 or 1
    :return: binary cross-entropy loss between y and y_hat
    """
    # raise Warning("You must implement binary_cross_entropy!")
    return -np.sum(np.multiply(y, np.log(y_hat + 1e-40)) + (np.multiply(1 - y, np.log(1 - y_hat + 1e-40))))

def sigmoid(x):
    """
    Compute sigmoid function on x, elementwise
    :param x: array of inputs [m, n]
    :return: array of outputs [m, n]
    """
    # raise Warning("You must implement sigmoid!")
    return (1/(1+np.exp(-x)))

def accuracy(y_pred, y):
    """
    Compute accuracy of predictions y_pred based on ground truth values y
    :param y_pred: Predicted values, THRESHOLDED TO 0 or 1, not probabilities
    :param y: Ground truth (target) values, also 0 or 1
    :return: Accuracy (scalar) 0.0 to 1.0
    """
    # raise Warning("You must implement accuracy!")
    return (np.max(y.shape) - np.count_nonzero(y_pred - y)) / np.max(y.shape)


def calculate_batches(X, batch_size):
    """
    Already implemented, don't worry about it
    :param X:
    :param batch_size:
    :return:
    """
    indices = list(range(X.shape[0]))
    np.random.shuffle(indices)
    args = [iter(indices)] * batch_size
    batch_indices = itertools.zip_longest(*args, fillvalue=None)
    return [list(j for j in i if j is not None) for i in batch_indices]

class LogisticRegression(object):
    def __init__(self, input_dimensions=2, seed=1234):
        """
        Initialize a Logistic Regression model
        :param input_dimensions: The number of features of the input data, for example (height, weight) would be two features.
        :param seed: Random seed for controlling/repeating experiments
        """
        np.random.seed(seed)
        self.input_dimensions = input_dimensions
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initialize weights of a logistic regression model using random numbers
        """
        self.weights = np.random.rand(self.input_dimensions+1,1)
        # raise Warning("You must implement _initialize_weights! This function should initialize (or re-initialize) your model weights. Use 'the bias trick' for this assignment")

    def fit(self, X_train, y_train, X_val, y_val, num_epochs=10, batch_size=4, alpha=0.01, _lambda=0.0):
        """
        Stochastic Gradient Descent training loop for a logistic regression model
        :param X_train: Training input data [n_samples, n_features+1], assume last column is 1's for the bias
        :param y_train: Training target values [n_samples, 1]
        :param X_val: Validation input data [n_samples, n_features+1], assume last column is 1's for the bias
        :param y_val: Validation target values [n_samples, 1]
        :param num_epochs: Number of training epochs (default 10)
        :param batch_size: Number of samples in a single batch (default 16)
        :param alpha: Learning rate for gradient descent update (default 0.01)
        :param _lambda: Coefficient for L2 weight regularization (default 0.0)
        :return: (train_error_, val_error) two arrays of cross-entropy values for each epoch (one for training and validation error)
        """
        # raise Warning("You must implement fit! This function should implement a mini-batch stochastic gradient descent training loop")
        train_xent = [] # append your cross-entropy on training set to this after each epoch
        val_xent = [] # append your cross-entropy on validation set to this after each epoch
        batch_indices = calculate_batches(X_train, batch_size)

        for epoch in range(num_epochs):
            for batch in batch_indices:
                self._train_on_batch(X_train[batch], y_train[batch], alpha, _lambda)
            train_xent.append(binary_cross_entropy(self.predict(X_train), y_train))
            val_xent.append(binary_cross_entropy(self.predict(X_val), y_val))
        return (train_xent, val_xent)

    def predict_proba(self, X):
        """
        Make a prediction on an array of inputs, must already contain bias as last column
        :param X: Array of input [n_samples, n_features+1]
        :return: Array of model outputs [n_samples, 1]. Each entry is a probability between 0 and 1
        """
        # raise Warning("You must implement predict_proba. This function should make a prediction on a batch (matrix) of inputs. The output should be probabilities.")
        return sigmoid(np.dot(X, self.weights))

    def predict(self, X):
        """
        Make a prediction on an array of inputs, and choose the nearest class, must already contain bias as last column
        :param X: Array of input [n_samples, n_features+1]
        :return: Array of model outputs [n_samples, 1]. Each entry is class ID (0 or 1)
        """
        # raise Warning("You must implement predict. This function should make a prediction on a batch (matrix) of inputs. The output should be class ID (0 or 1)")
        return np.around(sigmoid(np.dot(X, self.weights)))

    def _train_on_batch(self, X, y, alpha, _lambda):
        """
        Given a single batch of data, and the necessary hyperparameters, perform a single batch gradient update. This function should update the model weights.
        :param X: Batch of training input data [batch_size, n_features+1]
        :param y: Batch of training targets [batch_size, 1]
        :param alpha: Learning rate (scalar i.e. 0.01)
        :param _lambda: Regularization strength coefficient (scalar i.e. 0.0001)
        """
        gradients = self._binary_cross_entropy_gradient(X, y)
        momentum = np.abs(2*_lambda*gradients)
        self.weights = (self.weights - alpha*(gradients + momentum))
        self.weights = self._l2_regularization_gradient()
        # calculate output
        # calculate errors, binary cross entropy, and squared L2 regularization
        # calculate gradients of cross entropy and L2  w.r.t weights
        # perform gradient descent update
        # raise Warning("You must implement train on batch. This function should perform a stochastic gradient descent update on a single batch of samples")

    def _binary_cross_entropy_gradient(self, X, y):
        """
        Compute gradient of binary cross-entropy objective w.r.t model weights.
        :param X: Set of input data [n_samples, n_features+1]
        :param y: Set of target values [n_samples, 1]
        :return: Gradient of cross-entropy w.r.t model weights [n_features+1, 1]
        """
        # implement the binary cross-entropy gradient for logistic regression
        # this is the gradient W.R.T the weights
        y_hat = self.predict_proba(X)
        # raise Warning("You must implement the binary cross entropy gradient")
        return (np.dot((y_hat - y).T, X)/np.max(y.shape)).T

    def _l2_regularization_gradient(self):
        """
        Compute gradient for l2 weight regularization
        :return: Gradient of squared l2 norm w.r.t model weights [n_features+1, 1]
        """
        # raise Warning("You must implement the gradient for the squared l2 norm ")
        return self.weights

if __name__ == "__main__":
    print("This is library code. You may implement some functions here to do your own debugging if you want, but they won't be graded")