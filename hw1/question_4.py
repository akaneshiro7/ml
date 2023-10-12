import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Parameters
mu1 = 0
sigma1 = 1
mu2 = 1
sigma2 = np.sqrt(2)
prior_w1 = 0.5
prior_w2 = 0.5  # Assuming equal priors for simplicity

# Generate x values
x = np.linspace(-5, 5, 1000)

# Class conditional pdfs
pdf_w1 = norm.pdf(x, mu1, sigma1)
pdf_w2 = norm.pdf(x, mu2, sigma2)

# Plot class conditional pdfs
plt.figure()
plt.plot(x, pdf_w1, 'b-', label='p(x|w_1)')
plt.plot(x, pdf_w2, 'r-', label='p(x|w_2)')
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.title('Class Conditional PDFs')
plt.legend()
plt.show()

# Calculate posterior probabilities using Bayes' theorem
posterior_w1 = (pdf_w1 * prior_w1) / (pdf_w1 * prior_w1 + pdf_w2 * prior_w2)
posterior_w2 = (pdf_w2 * prior_w2) / (pdf_w1 * prior_w1 + pdf_w2 * prior_w2)

# Plot posterior probabilities
plt.figure()
plt.plot(x, posterior_w1, 'b-', label='p(w_1|x)')
plt.plot(x, posterior_w2, 'r-', label='p(w_2|x)')
plt.xlabel('x')
plt.ylabel('Posterior Probability')
plt.title('Posterior Probabilities')

# Decision boundary: Where the two posteriors are equal
idx = np.argmin(np.abs(posterior_w1 - posterior_w2))
decision_boundary = x[idx]
print(decision_boundary)
plt.axvline(decision_boundary, color='g', linestyle='--', label='Decision Boundary')
plt.legend()
plt.show()
