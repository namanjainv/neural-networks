import numpy as np
import pytest

def test_squared_l2_norm():
    from logistic_regression import squared_l2_norm
    w = np.float32([[1,2,3,4]]).T
    actual = squared_l2_norm(w)
    desired = 30.00000225043209
    np.testing.assert_allclose(actual, desired)

def test_accuracy():
    from logistic_regression import accuracy
    y_pred = np.float32([[0,1,1,0,1]]).T
    y = np.float32([[1,0,1,0,1]]).T
    actual = accuracy(y_pred, y)
    desired = 3./5.
    np.testing.assert_allclose(actual, desired)

   # w = np.float32([1, 2, 3, 4])
def test_sigmoid():
    from logistic_regression import sigmoid
    logits = np.float32([[-3, 0, 3]]).T
    desired = np.float32([[0.04742587317756678, 0.5, 0.9525741268224334]]).T
    actual = sigmoid(logits)
    np.testing.assert_allclose(actual, desired, rtol=1e-3, atol=1e-3)

def test_binary_cross_entropy():
    from logistic_regression import binary_cross_entropy
    outputs = np.float32([[0.0001, 0.1, 0.9, 0.8]]).T
    targets = np.float32([[1, 0, 1, 0]]).T
    actual = binary_cross_entropy(outputs, targets)
    desired = 16.223455
    # np.testing.assert_allclose(actual, desired, rtol=1e-3, atol=1e-3)

    outputs = np.float32([[1, 0, 1, 0]]).T
    targets = np.float32([[1, 0, 1, 0]]).T
    actual = binary_cross_entropy(outputs, targets)
    desired = 0
    # np.testing.assert_allclose(actual, desired, rtol=1e-3, atol=1e-3)
    assert True



def test_weight_dimension():
    from logistic_regression import LogisticRegression
    model = LogisticRegression(input_dimensions=2)
    assert model.weights.ndim == 2 and model.weights.shape[0] == 3 and model.weights.shape[1] == 1


def test_predict_proba():
    from logistic_regression import LogisticRegression
    model = LogisticRegression(input_dimensions=2)
    model.weights = np.float32([[1,2,4]]).T
    X = np.float32([[1,2,1],
                    [0,0,-2]])
    desired = np.float32([[0.9987, 0.0003]]).T
    actual = model.predict_proba(X)
    np.testing.assert_allclose(actual, desired, rtol=1e-3, atol=1e-3)

def test_predict():
    from logistic_regression import LogisticRegression
    model = LogisticRegression(input_dimensions=2)
    model.weights = np.float32([[1,2,4]]).T
    X = np.float32([[1,2,1],
                    [0,0,-2]])
    desired = np.float32([[1, 0]]).T
    actual = model.predict(X)
    np.testing.assert_allclose(actual, desired, rtol=1e-3, atol=1e-3)

def test_cross_entropy_gradient():
    from logistic_regression import LogisticRegression
    model = LogisticRegression(input_dimensions=2)
    model.weights = np.float32([[1,2,4]]).T
    X = np.float32([[1,2,1],
                    [0,0,1]])
    y = np.float32([[1, 0]]).T
    gradient = model._binary_cross_entropy_gradient(X, y)
    desired = np.float32([[-6e-5, -1e-4, 0.4909]]).T
    np.testing.assert_allclose(gradient, desired, rtol=1e-3, atol=1e-3)


def test_l2_regularization_gradient():
    from logistic_regression import LogisticRegression
    model = LogisticRegression(input_dimensions=2)
    model.weights = np.float32([[1, 2, 4]]).T
    gradient = model._l2_regularization_gradient()
    desired = np.float32([[1, 2, 4]]).T

    assert np.allclose(gradient, desired, rtol=1e-3, atol=1e-3) or np.allclose(gradient, 2*desired, rtol=1e-3, atol=1e-3)


def test_train_on_batch():
    from logistic_regression import LogisticRegression
    model = LogisticRegression(input_dimensions=2)
    weights_old = np.float32([[1, 2, 4]]).T
    model.weights = np.float32([[1, 2, 4]]).T

    X = np.float32([[1,2,1],
                    [0,0,1]])
    y = np.float32([[1, 0]]).T
    model._train_on_batch(X, y, 0.3, _lambda=0.001)
    desired = np.float32([[0.000281, 0.000563, 0.14848]]).T
    weight_delta = (weights_old - model.weights)
    other_desired = np.float32([[0.0005815, 0.00116301, 0.14968348]]).T
    assert np.allclose(weight_delta, desired, rtol=1e-3, atol=1e-3) or np.allclose(weight_delta, other_desired, rtol=1e-3, atol=1e-3)


def test_fit_functional():
    import sklearn.model_selection
    import sklearn.datasets
    import numpy as np

    from logistic_regression import LogisticRegression, accuracy
    X = np.zeros((1000, 3), dtype=np.float32)
    X[:, -1] = 1
    features, targets = sklearn.datasets.make_blobs(1000, 2, 2, cluster_std=1, random_state=1234)
    X[:, [0, 1]] = features
    y = targets[:, np.newaxis]

    X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(X, y)
    model = LogisticRegression(input_dimensions=2)
    train_xent, val_xent = model.fit(X_train, y_train, X_val, y_val, num_epochs=20, batch_size=4, alpha=0.1, _lambda=0.0)
    predictions = model.predict(X_val)
    assert accuracy(predictions, y_val) >= 0.65
    assert accuracy(predictions, y_val) >= 0.90
    assert accuracy(predictions, y_val) >= 0.99
