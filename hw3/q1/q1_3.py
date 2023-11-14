import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit 

np.random.seed(0)

def logistic_function(theta, X):
    return expit(np.dot(X, theta))

def negative_log_likelihood(theta, X, y):
    h = logistic_function(theta, X)
    return -np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))

def classify(theta, X):
    probabilities = logistic_function(theta, X)
    return (probabilities >= 0.5).astype(int)

def logistic_quadratic_function(theta, X):
    return expit(np.dot(X, theta))

def negative_log_likelihood_quadratic(theta, X, y):
    h = logistic_quadratic_function(theta, X)
    return -np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))

def add_quadratic_terms(X):
    quadratic_terms = X**2
    return np.hstack((np.ones((X.shape[0], 1)), X, quadratic_terms))

for f in ['d_100_train.csv', 'd_1000_train.csv', 'd_10000_train.csv']:
    print(f'----------------Using {f}-------------------' )
    train_data = pd.read_csv(f'hw3/q1/{f}')
    validate_data = pd.read_csv('hw3/q1/d_20000_validate.csv')

    train_data['Label'] = train_data['Label'].map({'L0': 0, 'L1': 1})
    validate_data['Label'] = validate_data['Label'].map({'L0': 0, 'L1': 1})

    # Separate features and target variable
    X_train = train_data[['X', 'Y']]
    y_train = train_data['Label']
    X_validate = validate_data[['X', 'Y']]
    y_validate = validate_data['Label']


    X_train_with_intercept = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
    X_validate_with_intercept = np.hstack((np.ones((X_validate.shape[0], 1)), X_validate))

    initial_theta = np.zeros(X_train_with_intercept.shape[1])

    result = minimize(negative_log_likelihood, initial_theta, args=(X_train_with_intercept, y_train))

    theta_optimized = result.x


    y_pred_validate = classify(theta_optimized, X_validate_with_intercept)

    error_count = np.sum(y_pred_validate != y_validate)
    total_samples = len(y_validate)
    probability_of_error = error_count / total_samples

    print(f"Probability of Error Linear: {probability_of_error}")


    X_train_quadratic = add_quadratic_terms(X_train)
    X_validate_quadratic = add_quadratic_terms(X_validate)

    initial_theta_quadratic = np.zeros(X_train_quadratic.shape[1])

    result_quadratic = minimize(negative_log_likelihood_quadratic, initial_theta_quadratic, args=(X_train_quadratic, y_train))

    theta_optimized_quadratic = result_quadratic.x

    y_pred_validate_quadratic = classify(theta_optimized_quadratic, X_validate_quadratic)

    error_count_quadratic = np.sum(y_pred_validate_quadratic != y_validate)
    probability_of_error_quadratic = error_count_quadratic / total_samples

    print(f"Probability of Error Quadratic: {probability_of_error_quadratic}")
