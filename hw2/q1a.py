import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# Given Values
m01 = [3, 0]  
C01 = [[2, 0], [0, 1]]
m02 = [0, 3]  
C02 = [[1, 0], [0, 2]]
m1 = [2, 2]  
C1 = [[1, 0], [0, 1]]
w1 = 0.65
w2 = 0.35
n_samples = 10000

# Read CSV
samples_L0 = pd.read_csv("samples_L0.csv")
samples_L1 = pd.read_csv("samples_L1.csv")

# Define Classifier based on Gamma
def classify(x, y, gamma):
    L01 = multivariate_normal.pdf(np.array([x, y]).T, mean=m01, cov=C01)
    L02 = multivariate_normal.pdf(np.array([x, y]).T, mean=m02, cov=C02)
    L0 = (L01 + L02) / 2
    L1 = multivariate_normal.pdf(np.array([x, y]).T, mean=m1, cov=C1)
    
    return (L1 / L0) > gamma

# Define Gamma Values
gammas = np.concatenate([np.array([0, 0.01, 0.1, 0.25]), np.arange(0.5, 5, 0.01), np.arange(5, 100, 1), np.array([1000, 10000, float("inf")])])

# Apply Get Decisions for each gamma for each sample 
for gamma in gammas:
    samples_L0["Decision: g=" + str(gamma)] = classify(samples_L0["x"], samples_L0["y"], gamma)
    samples_L1["Decision: g=" + str(gamma)] = classify(samples_L1["x"], samples_L1["y"], gamma)

# True Positive rate, False Postive rate
tpr = []
fpr = []

theoretically_optimal_gamma = [None, float("inf"), 0, 0]

# Get Distance of tpr and fpr from (0, 1)
def getDistance(tpr, fpr):
    return np.sqrt((fpr - 0) ** 2 + (tpr - 1) ** 2)

# Get theo optimal gamma up to 0.01 Decimals by looping through array of 0.5 - 5 in steps of 0.01
for gamma in gammas:
    truePositiveRate = (samples_L1['Decision: g=' + str(gamma)] == 1).sum() / len(samples_L1)
    falsePositiveRate = (samples_L0['Decision: g=' + str(gamma)] == 1).sum() / len(samples_L0)
    tpr.append(truePositiveRate)
    fpr.append(falsePositiveRate)
    distance = getDistance(truePositiveRate, falsePositiveRate)
    if distance < theoretically_optimal_gamma[1]:
        theoretically_optimal_gamma = [gamma, distance, truePositiveRate, falsePositiveRate]

# Calculate Error
theoreticalErrorMinimum = (1 - theoretically_optimal_gamma[2]) * w2 + theoretically_optimal_gamma[3] * w1

plt.figure(figsize=(10, 7))
plt.plot(fpr, tpr, linestyle='-', color='b')
plt.plot([0, 1], [0, 1], linestyle='--', color='k')

plt.scatter(theoretically_optimal_gamma[3], theoretically_optimal_gamma[2], color="red", marker='x', s=100, label=f'Theoretical Optimal Point\nGamma={theoretically_optimal_gamma[0]:.2f}, P(error)={theoreticalErrorMinimum:.2f}')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.grid(True)

min_p_error = float('inf')
optimal_gamma = None

# get empirical optimal gamma
for gamma, tpr_value, fpr_value in zip(gammas, tpr, fpr):
    p_error = w1 * fpr_value + w2 * (1 - tpr_value) 
    if p_error < min_p_error:
        min_p_error = p_error
        optimal_gamma = gamma

# Marking the operating point on the ROC curve
optimal_fpr = fpr[gammas.tolist().index(optimal_gamma)]
optimal_tpr = tpr[gammas.tolist().index(optimal_gamma)]
plt.scatter(optimal_fpr, optimal_tpr, color="green", marker='o', s=100, label=f'Empirical Optimal Point\nGamma={optimal_gamma:.2f}, P(error)={min_p_error:.2f}')
plt.legend()

plt.show()

