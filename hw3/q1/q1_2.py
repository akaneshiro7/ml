import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal

np.random.seed(0)

for f in ['d_100_train.csv', 'd_1000_train.csv', 'd_10000_train.csv']:
    print(f'----------------Using {f}-------------------' )
    # Read Data
    train_data = pd.read_csv(f'hw3/q1/{f}')
    validation_data = pd.read_csv('hw3/q1/d_20000_validate.csv')

    # Estimate Priors
    prior_L0 =len(train_data[train_data['Label'] == 'L0']) / len(train_data)
    prior_L1 =len(train_data[train_data['Label'] == 'L1']) / len(train_data)

    # Preprocess Data
    data_L0 = train_data[train_data['Label'] == 'L0'][['X', 'Y']].values
    data_L1 = train_data[train_data['Label'] == 'L1'][['X', 'Y']]


    # Get random initial mean, covariances, and weights
    def initialize_parameters(data):
        n, d = data.shape
        means = data[np.random.choice(n, 2, replace=False)]
        covariances = np.array([np.cov(data, rowvar=False)] * 2)
        weights = np.array([0.5, 0.5])
        return means, covariances, weights

    # Exxpectation Step
    def expectation(data, means, covariances, weights):
        n, d = data.shape
        k = len(weights)
        responsibilities = np.zeros((n, k))
        
        for i in range(k):
            distribution = multivariate_normal(mean=means[i], cov=covariances[i])
            responsibilities[:, i] = weights[i] * distribution.pdf(data)
        
        responsibilities_sum = np.sum(responsibilities, axis=1)[:, np.newaxis]
        responsibilities /= responsibilities_sum
        
        return responsibilities

    # Maximization Step
    def maximization(data, responsibilities):
        n, d = data.shape
        k = responsibilities.shape[1]
        
        nk = np.sum(responsibilities, axis=0)
        weights = nk / n
        means = np.dot(responsibilities.T, data) / nk[:, np.newaxis]
        covariances = np.zeros((k, d, d))
        
        for i in range(k):
            diff = data - means[i]
            covariances[i] = np.dot(responsibilities[:, i] * diff.T, diff) / nk[i]
        
        return means, covariances, weights

    # Get Estimated mean, covariances, weights for gaussian mixture of 2 components.
    def gmm(data, n_iter=100):
        means, covariances, weights = initialize_parameters(data)
        
        for _ in range(n_iter):
            responsibilities = expectation(data, means, covariances, weights)
            means, covariances, weights = maximization(data, responsibilities)
        
        return means, covariances, weights
    

    means_L0, covariances_L0, weights_L0 = gmm(data_L0)

    mean_L1 = data_L1.mean()
    variance_L1 = data_L1.var()

    # for i in range(2):
    #     print(f"Mean L0{i}: {means_L0[i]}")
    #     print(f"Covariance L0{i}: {covariances_L0[i]}")
    #     print(f"Weights L0{i}: {weights_L0[i]}")

    # print(f"Mean L1: {mean_L1.X, mean_L1.Y}")
    # print(f"Covariance L1: {variance_L1.X, variance_L1.Y}")

    # Calculate TPR and FPR
    tpr = []
    fpr = []

    gammas = np.concatenate([np.array([0, 0.01, 0.1, 0.25]), np.arange(0.5, 1.4, 0.01), np.array([1.5]), np.arange(1.4, 5, 0.01),np.arange(5, 100, 1), np.array([1000, 10000, float("inf")])])

    def classify(x, y, gamma):
        L01 = multivariate_normal.pdf(np.array([x, y]).T, mean=means_L0[0], cov=covariances_L0[0])
        L02 = multivariate_normal.pdf(np.array([x, y]).T, mean=means_L0[1], cov=covariances_L0[0])
        L0 = L01 * weights_L0[0] + L02 * weights_L0[1] 
        L1 = multivariate_normal.pdf(np.array([x, y]).T, mean=mean_L1, cov=variance_L1)
        
        return (L1 / L0) > gamma 

    # Loop through gammas
    for gamma in gammas:
        validation_data['Decision: g= ' + str(gamma)] = classify(validation_data['X'], validation_data['Y'], gamma) 

    # Get true posoitive and false positives
    for gamma in gammas:
        true_positives = validation_data[(validation_data['Label'] == 'L1') & (validation_data['Decision: g= ' + str(gamma)] == True)].shape[0] / len(validation_data[validation_data['Label'] == 'L1'])
        false_positives = validation_data[(validation_data['Label'] != 'L1') & (validation_data['Decision: g= ' + str(gamma)] == True)].shape[0] / len(validation_data[validation_data['Label'] != 'L1'])
        tpr.append(true_positives)
        fpr.append(false_positives)

    # Plot
    plt.figure(figsize=(10, 7))
    plt.plot(fpr, tpr, linestyle='-', color='b')
    plt.plot([0, 1], [0, 1], linestyle='--', color='k')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.grid(True)

    # Get Optimal Gamma
    optimal_gamma = round(prior_L0 / prior_L1, 1)

    # Get optimal tpr and fpr
    optimal_tpr = validation_data[(validation_data['Label'] == 'L1') & (validation_data[f'Decision: g= {optimal_gamma}'] == True)].shape[0] / len(validation_data[validation_data['Label'] == 'L1'])
    optimal_fpr = validation_data[(validation_data['Label'] != 'L1') & (validation_data[f'Decision: g= {optimal_gamma}'] == True)].shape[0] / len(validation_data[validation_data['Label'] != 'L1'])

    # calculate min p_error
    p_error = weights_L0[1] * optimal_fpr + weights_L0[0] * (1 - optimal_tpr) 

    plt.scatter(optimal_fpr, optimal_tpr, color="red", marker="x", s=100, label=f"Theoretical Optimal Point\nGamma=1.5\nP(error)={p_error:.2f}")
    plt.legend()
    plt.show()

    print(f"P(error) = {p_error}")