import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# Given parameters
m01 = [3, 0]  # Flattened to 1D
C01 = [[2, 0], [0, 1]]
m02 = [0, 3]  # Flattened to 1D
C02 = [[1, 0], [0, 2]]
m1 = [2, 2]   # Flattened to 1D
C1 = [[1, 0], [0, 1]]
w1 = 0.65
w2 = 0.35
n_samples = 10000

samples_L0 = pd.read_csv("samples_L0.csv")
samples_L1 = pd.read_csv("samples_L1.csv")

# # Classifier based on likelihood ratio
def classify(x, gamma):
    L01 = multivariate_normal.pdf(x, mean=m01, cov=C01)
    L02 = multivariate_normal.pdf(x, mean=m02, cov=C02)

    L0 = (L01 + L02) / 2
    L1 = multivariate_normal.pdf(x, mean=m1, cov=C1)

    return 1 if (L1 / L0) > gamma else 0
 
gammas = [0, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10_000, float("inf")]

for gamma in gammas:
    samples_L0["Decision: g=" + str(gamma)] = samples_L0.apply(lambda row: classify([row["x"], row["y"]], gamma), axis=1)
    samples_L1["Decision: g=" + str(gamma)] = samples_L1.apply(lambda row: classify([row["x"], row["y"]], gamma), axis=1)


tpr = []
fpr = []

for gamma in gammas:
    truePositiveRate = (samples_L1['Decision: g=' + str(gamma)] == 1).sum() / len(samples_L1)
    falsePositiveRate = (samples_L0['Decision: g=' + str(gamma)] == 1).sum() / len(samples_L0)
    tpr.append(truePositiveRate)
    fpr.append(falsePositiveRate)


plt.figure(figsize=(10, 7))
plt.plot(fpr, tpr, marker='o', linestyle='-', color='b')
plt.plot([0, 1], [0, 1], linestyle='--', color='k')

for i, g in enumerate(gammas):
    plt.annotate(f'{g}', (fpr[i], tpr[i]), textcoords="offset points", xytext=(0,10), ha='center')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.grid(True)
plt.show()

