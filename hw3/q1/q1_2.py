import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal

train_data = pd.read_csv('hw3/q1/d_10000_train.csv')
validation_data = pd.read_csv('hw3/q1/d_20000_validate.csv')

prior_L0 =len(train_data[train_data['Label'] == 'L0']) / len(train_data)
prior_L1 =len(train_data[train_data['Label'] == 'L1']) / len(train_data)

data_L0 = train_data[train_data['Label'] == 'L0'][['X', 'Y']]
data_L1 = train_data[train_data['Label'] == 'L1'][['X', 'Y']]

gmm_L0 = GaussianMixture(n_components=2, covariance_type='full')
gmm_L0.fit(data_L0)

gnb_L1 = GaussianNB()
gnb_L1.fit(data_L1, np.zeros(len(data_L1)))  

means_L0 = gmm_L0.means_
covariances_L0 = gmm_L0.covariances_
weights_L0 = gmm_L0.weights_

mean_L1 = gnb_L1.theta_[0]
variance_L1 = gnb_L1.var_[0]

tpr = []
fpr = []

gammas = np.concatenate([np.array([0, 0.01, 0.1, 0.25]), np.arange(0.5, 1.4, 0.01), np.array([1.5]), np.arange(1.4, 5, 0.01),np.arange(5, 100, 1), np.array([1000, 10000, float("inf")])])

def classify(x, y, gamma):
    L01 = multivariate_normal.pdf(np.array([x, y]).T, mean=means_L0[0], cov=covariances_L0[0])
    L02 = multivariate_normal.pdf(np.array([x, y]).T, mean=means_L0[1], cov=covariances_L0[0])
    L0 = L01 * weights_L0[0] + L02 * weights_L0[1] 
    L1 = multivariate_normal.pdf(np.array([x, y]).T, mean=mean_L1, cov=variance_L1)
    
    return (L1 / L0) > gamma 

for gamma in gammas:
    train_data['Decision: g= ' + str(gamma)] = classify(train_data['X'], train_data['Y'], gamma) 


gammas = np.concatenate([np.array([0, 0.01, 0.1, 0.25]), np.arange(0.5, 1.4, 0.01), np.array([1.5]), np.arange(1.4, 5, 0.01),np.arange(5, 100, 1), np.array([1000, 10000, float("inf")])])

for gamma in gammas:
    validation_data['Decision: g= ' + str(gamma)] = classify(validation_data['X'], validation_data['Y'], gamma) 


for gamma in gammas:
    true_positives = validation_data[(validation_data['Label'] == 'L1') & (validation_data['Decision: g= ' + str(gamma)] == True)].shape[0] / len(validation_data[validation_data['Label'] == 'L1'])
    false_positives = validation_data[(validation_data['Label'] != 'L1') & (validation_data['Decision: g= ' + str(gamma)] == True)].shape[0] / len(validation_data[validation_data['Label'] != 'L1'])
    tpr.append(true_positives)
    fpr.append(false_positives)

plt.figure(figsize=(10, 7))
plt.plot(fpr, tpr, linestyle='-', color='b')
plt.plot([0, 1], [0, 1], linestyle='--', color='k')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.grid(True)

optimal_gamma = round(prior_L0 / prior_L1, 1)

optimal_tpr = validation_data[(validation_data['Label'] == 'L1') & (validation_data[f'Decision: g= {optimal_gamma}'] == True)].shape[0] / len(validation_data[validation_data['Label'] == 'L1'])
optimal_fpr = validation_data[(validation_data['Label'] != 'L1') & (validation_data[f'Decision: g= {optimal_gamma}'] == True)].shape[0] / len(validation_data[validation_data['Label'] != 'L1'])

p_error = weights_L0[1] * optimal_fpr + weights_L0[0] * (1 - optimal_tpr) 

plt.scatter(optimal_fpr, optimal_tpr, color="red", marker="x", s=100, label=f"Theoretical Optimal Point\nGamma=1.5\nP(error)={p_error:.2f}")
plt.legend()
plt.show()

print(p_error)