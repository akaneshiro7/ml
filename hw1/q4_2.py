import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

mu1, sigma1 = 0, 1
mu2, sigma2 = 1, np.sqrt(2)

x = np.linspace(-5, 5, 1000)

# Class conditional pdfs
p_x_w1 = norm.pdf(x, mu1, sigma1)
p_x_w2 = norm.pdf(x, mu2, sigma2)

# Calculate posterior probabilities using Bayes' theorem
prior_w1, prior_w2 = 0.5, 0.5
p_x = p_x_w1 * prior_w1 + p_x_w2 * prior_w2
p_w1_x = (p_x_w1 * prior_w1) / p_x
p_w2_x = (p_x_w2 * prior_w2) / p_x

# Decision boundary formula
def getDecisionBoundaries(mu, sigma):
    discriminant = np.sqrt((4 * (mu ** 2)) - 4 * (-1 * sigma ** 2 + 1) * (mu **2 + sigma ** 2 * np.log(sigma**2)) )
    return [((2 * mu + discriminant) / (2 * (-sigma ** 2 + 1))), ((2 * mu - discriminant) / (2 * (-sigma ** 2 + 1)))]

decision_boundaries = getDecisionBoundaries(mu2, sigma2)

# Plot class conditional pdfs
plt.figure(figsize=(10, 5))
plt.plot(x, p_x_w1, label='p(x|w1)')
plt.plot(x, p_x_w2, label='p(x|w2)')
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.title('Class Conditional PDFs')
plt.legend()
plt.grid(True)

for boundary in decision_boundaries:
    plt.axvline(x=boundary, color='red', linestyle='--', label='Decision Boundary')
    plt.annotate(f'{boundary:.2f}', (boundary, 0.05), textcoords="offset points", xytext=(0,10), ha='right', color='red')
plt.show()

# Plot posterior probabilities
plt.figure(figsize=(10, 5))
plt.plot(x, p_w1_x, label='p(w1|x)')
plt.plot(x, p_w2_x, label='p(w2|x)')
plt.xlabel('x')
plt.ylabel('Probability')
plt.title('Posterior Probabilities')
plt.legend()
plt.grid(True)
for boundary in decision_boundaries:
    plt.axvline(x=boundary, color='red', linestyle='--', label='Decision Boundary')
    plt.annotate(f'{boundary:.2f}', (boundary, 0.05), textcoords="offset points", xytext=(0,10), ha='right', color='red')
plt.show()