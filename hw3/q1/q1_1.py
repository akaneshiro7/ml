import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

prior_0 = 0.6
prior_1 = 0.4

m01 = [5, 0]
c01 = [[4, 0], [0, 2]]
m02 = [0, 4]
c02 = [[1, 0], [0, 3]]
w1 = w2 = 0.5

m1 = [3, 2]
c1 = [[2, 0], [0, 2]]

d_20K = pd.read_csv('hw3/q1/d_20000_validate.csv')

# Calculated Optimal
theoretically_optimal_gamma = 1.5
# Gamma range
gammas = np.concatenate([np.array([0, 0.01, 0.1, 0.25]), np.arange(0.5, 1.4, 0.01), np.array([1.5]), np.arange(1.4, 5, 0.01),np.arange(5, 100, 1), np.array([1000, 10000, float("inf")])])

# True Positive rate, False Postive rate
tpr = []
fpr = []

# Classification function
def classify(x, y, gamma):
    L01 = multivariate_normal.pdf(np.array([x, y]).T, mean=m01, cov=c01)
    L02 = multivariate_normal.pdf(np.array([x, y]).T, mean=m02, cov=c02)
    L0 = L01 * w1 + L02 * w2
    L1 = multivariate_normal.pdf(np.array([x, y]).T, mean=m1, cov=c1)
    
    return (L1 / L0) > gamma 

# Classify based on gammas
for gamma in gammas:
    d_20K['Decision: g= ' + str(gamma)] = classify(d_20K['X'], d_20K['Y'], gamma) 

# Calculate false and true postiive rates
for gamma in gammas:
    true_positives = d_20K[(d_20K['Label'] == 'L1') & (d_20K['Decision: g= ' + str(gamma)] == True)].shape[0] / len(d_20K[d_20K['Label'] == 'L1'])
    false_positives = d_20K[(d_20K['Label'] != 'L1') & (d_20K['Decision: g= ' + str(gamma)] == True)].shape[0] / len(d_20K[d_20K['Label'] != 'L1'])
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

optimal_tpr = d_20K[(d_20K['Label'] == 'L1') & (d_20K['Decision: g= 1.5'] == True)].shape[0] / len(d_20K[d_20K['Label'] == 'L1'])
optimal_fpr = d_20K[(d_20K['Label'] != 'L1') & (d_20K['Decision: g= 1.5'] == True)].shape[0] / len(d_20K[d_20K['Label'] != 'L1'])
p_error = w1 * optimal_fpr + w2 * (1 - optimal_tpr) 

plt.scatter(optimal_fpr, optimal_tpr, color="red", marker="x", s=100, label=f"Theoretical Optimal Point\nGamma=1.5\nP(error)={p_error:.2f}")
plt.legend()
plt.show()

print(f"Min_p_error = {p_error}")