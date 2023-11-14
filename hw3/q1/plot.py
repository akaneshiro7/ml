import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file into a DataFrame
df = pd.read_csv('./hw3/q1/d_20000_validate.csv')

# Separate data for labels L0 and L1
l0_data = df[df['Label'] == 'L0']
l1_data = df[df['Label'] == 'L1']

# Plot the points with different colors for L0 and L1
plt.scatter(l0_data['X'], l0_data['Y'], color='green', label='L0')
plt.scatter(l1_data['X'], l1_data['Y'], color='red', label='L1')

# Add labels and legend
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()

# Show the plot
plt.show()
