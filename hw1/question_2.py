import numpy as np
import matplotlib.pyplot as plt

# Given values
a1, b1 = 0, 1
a2, b2 = 1, 2

# Define the densities
def p(x, a, b):
    return (1 / (2 * b)) * np.exp(-np.abs(x - a) / b)

# Define the likelihood ratio
def likelihood_ratio(x):
    return p(x, a1, b1) / p(x, a2, b2)

# Generate x values
x = np.linspace(-5, 5, 400)

# Compute the likelihood ratio for each x
lr = likelihood_ratio(x)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(x, lr, label=r'$\Lambda(x) = \frac{p(x|w_1)}{p(x|w_2)}$')
plt.title('Likelihood Ratio for a=1,b=1,a2=1,b=2')
plt.xlabel('x')
plt.ylabel('Likelihood Ratio')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
