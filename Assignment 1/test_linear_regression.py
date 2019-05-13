import numpy as np
import pytest

def test_squared_l2_norm():
    from linear_regression import squared_l2_norm
    w = np.float32([[1,2,3,4]]).T
    actual = squared_l2_norm(w)
    desired = 30.00000225043209
    np.testing.assert_allclose(actual, desired)

   # w = np.float32([1, 2, 3, 4])

def test_mean_squared_error():
    from linear_regression import mean_squared_error
    outputs = np.float32([[1, 2, 3, 4]]).T
    targets = np.float32([[1.1, 2.2, 3.3, 4.4]]).T
    actual = mean_squared_error(outputs, targets)
    desired = 0.07500000000000007
    np.testing.assert_allclose(actual, desired, rtol=1e-3, atol=1e-3)

def test_weight_dimension():
    from linear_regression import LinearRegression
    model = LinearRegression(input_dimensions=2)
    assert model.weights.ndim == 2 and model.weights.shape[0] == 3 and model.weights.shape[1] == 1


def test_predict():
    from linear_regression import LinearRegression
    model = LinearRegression(input_dimensions=2)
    model.weights = np.float32([[1,2,4]]).T
    X = np.float32([[1,2,1],
                    [0,0,1]])
    desired = np.float32([[9, 4]]).T
    actual = model.predict(X)
    np.testing.assert_allclose(actual, desired, rtol=1e-3, atol=1e-3)

def test_mse_gradient():
    from linear_regression import LinearRegression
    model = LinearRegression(input_dimensions=2)
    model.weights = np.float32([[1,2,4]]).T
    X = np.float32([[1,2,1],
                    [0,0,1]])
    y = np.float32([[10, 2]]).T
    gradient = model._mse_gradient(X, y)
    desired = np.float32([[-0.5, -1.,   0.5]]).T
    print(gradient)
    print(desired)
    assert np.allclose(gradient, desired, rtol=1e-3, atol=1e-3) or np.allclose(gradient, 2*desired, rtol=1e-3, atol=1e-3)

# Check
def test_l2_regularization_gradient():
    from linear_regression import LinearRegression
    model = LinearRegression(input_dimensions=2)
    model.weights = np.float32([[1, 2, 4]]).T
    gradient = model._l2_regularization_gradient()
    desired = np.float32([[1, 2, 4]]).T
    assert (np.allclose(gradient, desired, rtol=1e-3, atol=1e-3) or np.allclose(gradient, 2*desired, rtol=1e-3, atol=1e-3))

def test_train_on_batch():
    from linear_regression import LinearRegression
    model = LinearRegression(input_dimensions=2)
    weights_old = np.float32([[1, 2, 4]]).T
    model.weights = np.float32([[1, 2, 4]]).T

    X = np.float32([[1,2,1],
                    [0,0,1]])
    y = np.float32([[10, 2]]).T
    model._train_on_batch(X, y, 0.3, _lambda=0.001)
    desired = np.float32([[-0.14970, -0.29940, 0.15120]]).T
    weight_delta = (weights_old - model.weights)
    other_desired = np.float32([[-0.2994001, -0.5988002, 0.3024001]]).T
    assert np.allclose(weight_delta, desired, rtol=1e-3, atol=1e-3) or np.allclose(weight_delta, other_desired, rtol=1e-3, atol=1e-3)

def test_fit_functional():
    import sklearn.model_selection
    import numpy as np

    from linear_regression import LinearRegression
    X = np.zeros((900, 3), dtype=np.float32)
    num_samples = 30

    xx = np.linspace(-5, 5, num_samples)
    XX, YY = np.meshgrid(xx, xx)
    X[:, 0] = XX.flatten()
    X[:, 1] = YY.flatten()
    X[:, -1] = 1  # a column of 1's for the bias trick
    Z = 0.1 * XX + 0.2 * YY + 0.4
    y = Z.reshape(-1, 1)
    X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(X, y)
    model = LinearRegression(input_dimensions=2)
    train_mse, val_mse = model.fit(X_train, y_train, X_val, y_val, num_epochs=20, batch_size=4, alpha=0.1, _lambda=0.0)
    final_train_mse = train_mse[-1]
    #desired_weights = np.float32([[0.1, 0.2, 0.4]]).T
    #np.testing.assert_allclose(model.weights, desired_weights, rtol=1e-3, atol=1e-3)
    assert final_train_mse < 0.001
    assert final_train_mse < 0.00001
    assert final_train_mse < 1e-10


