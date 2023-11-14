import pandas as pd
import matplotlib.pyplot as plt

data_files = ["d_100_train.csv", "d_1000_train.csv", "d_10000_train.csv", "d_20000_validate.csv"]

for f in data_files:
    df = pd.read_csv(f'hw3/q1/{f}')

    l0_data = df[df['Label'] == 'L0']
    l1_data = df[df['Label'] == 'L1']

    plt.scatter(l0_data['X'], l0_data['Y'], color='green', label='L0')
    plt.scatter(l1_data['X'], l1_data['Y'], color='red', label='L1')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()

    plt.title(f"Plot of {f}")
    plt.show()
