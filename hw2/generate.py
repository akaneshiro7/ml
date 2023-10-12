import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

import csv

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

np.random.seed(0)

# Generate samples
samples_L01 = multivariate_normal.rvs(mean=m01, cov=C01, size=int(n_samples * w1 * 0.5))
samples_L02 = multivariate_normal.rvs(mean=m02, cov=C02, size=int(n_samples * w1 * 0.5))
samples_L0 = np.concatenate((samples_L01, samples_L02))
samples_L1 = multivariate_normal.rvs(mean=m1, cov=C1, size=int(n_samples * w2))

with open('samples_L0.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['x', 'y'])
    for sample in samples_L0:
        writer.writerow(sample)

with open('samples_L1.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['x', 'y'])
    for sample in samples_L1:
        writer.writerow(sample)

plt.scatter(samples_L0[:,0], samples_L0[:,1], label='L=0 Samples', color='red')
plt.scatter(samples_L1[:,0], samples_L1[:,1], label='L=1 Samples', color='blue')

plt.title("Scatter Plot of Data")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()

plt.show()
