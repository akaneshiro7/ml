import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import pandas as pd
np.random.seed(0)

prior_0 = 0.6
prior_1 = 0.4
m01 = [5, 0]
c01 = [[4, 0], [0, 2]]
m02 = [0, 4]
c02 = [[1, 0], [0, 3]]
w1 = w2 = 0.5
m1 = [3, 2]
c1 = [[2, 0], [0, 2]]

def generate_samples(size, output_file):
    size_L0 = int(prior_0 * size)
    size_L1 = int(prior_1 * size)
    samples_L01 = multivariate_normal.rvs(mean=m01, cov=c01, size=int(size_L0 * w1))
    samples_L02 = multivariate_normal.rvs(mean=m02, cov=c02, size=int(size_L0 * w2))
    samples_L0 = np.concatenate((samples_L01, samples_L02))

    samples_L1 = multivariate_normal.rvs(mean=m1, cov=c1, size=int(size_L1))

    labels_L0 = np.array(['L0'] * size_L0)
    labels_L1 = np.array(['L1'] * size_L1)

    all_labels = np.concatenate((labels_L0, labels_L1))
    all_samples = np.concatenate((samples_L0, samples_L1))

    df = pd.DataFrame(all_samples, columns=['X', 'Y'])
    df['Label'] = all_labels
    df.to_csv(output_file, index=False)

    return

generate_samples(100, 'd_100_train.csv')
generate_samples(1000, 'd_1000_train.csv')
generate_samples(10_000, 'd_10000_train.csv')
generate_samples(20_000, 'd_20000_validate.csv')
