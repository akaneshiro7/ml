import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
import pandas as pd

# Read values to df
samples_L0 = pd.read_csv("samples_L0.csv")
samples_L1 = pd.read_csv("samples_L1.csv")

# Estimate mean and cov
mu_1 = np.mean(samples_L1, axis=0)
mu_0 = np.mean(samples_L0, axis=0)
cov_1 = np.cov(samples_L1, rowvar=False)
cov_0 = np.cov(samples_L0, rowvar=False)

S_W = 0.5 * (cov_1 + cov_0)

w_LDA = np.linalg.inv(S_W).dot(mu_1 - mu_0)

# get projections
projected_L1 = np.dot(samples_L1, w_LDA)
projected_L0 = np.dot(samples_L0, w_LDA)

fpr, tpr, thresholds = roc_curve([0]*len(projected_L0) + [1]*len(projected_L1),
                                  np.concatenate((projected_L0, projected_L1)))

min_p_error = np.inf
min_p_error_threshold = None

# Get min prob
for t, f, tp in zip(thresholds, fpr, tpr):
    p_error = 0.5 * (f + (1 - tp)) 
    if p_error < min_p_error:
        min_p_error = p_error
        min_p_error_threshold = t

# Plot roc curve
plt.plot(fpr, tpr, label='ROC Curve')
plt.scatter(fpr[thresholds==min_p_error_threshold], tpr[thresholds==min_p_error_threshold], marker='x', color='r', label=f'Min P(error): {min_p_error} \n Threshold: {min_p_error_threshold}')
plt.plot([0, 1], [0, 1], linestyle='--', color='k')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve using LDA')
plt.legend()
plt.grid(True)
plt.show()

