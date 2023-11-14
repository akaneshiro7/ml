import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

np.random.seed(0)

# Parameters
measurement_noise_std = 0.3
sigma_x = sigma_y = 0.25
vehicle_true_position = np.random.rand(2) * 2 - 1  # True vehicle position in the unit circle

# Function to generate landmarks
def generate_landmarks(K):
    angles = np.linspace(0, 2 * np.pi, K, endpoint=False)
    landmarks = np.vstack((np.cos(angles), np.sin(angles))).T
    return landmarks

# Function to generate range measurements
def generate_measurements(landmarks, vehicle_position):
    distances = np.linalg.norm(landmarks - vehicle_position, axis=1)
    measurements = distances + np.random.randn(*distances.shape) * measurement_noise_std
    measurements = np.maximum(measurements, 0)  # Ensure non-negative measurements
    return measurements

# Function to calculate the MAP objective function value
def map_objective_function(x, y, landmarks, measurements):
    likelihood = np.sum([(r - np.hypot(x - lx, y - ly))**2 
                         for (lx, ly), r in zip(landmarks, measurements)])
    prior = multivariate_normal(mean=[0, 0], cov=[[sigma_x**2, 0], [0, sigma_y**2]])
    return -(likelihood / (2 * measurement_noise_std**2)) + np.log(prior.pdf([x, y]))

# Generate contour plots for different K values
x_values = np.linspace(-2, 2, 400)
y_values = np.linspace(-2, 2, 400)
X, Y = np.meshgrid(x_values, y_values)

for K in [1, 2, 3, 4]:
    landmarks = generate_landmarks(K)
    measurements = generate_measurements(landmarks, vehicle_true_position)
    Z = np.zeros_like(X)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = map_objective_function(X[i, j], Y[i, j], landmarks, measurements)
    
    plt.figure(figsize=(8, 6))
    contour = plt.contour(X, Y, Z, 20)
    plt.clabel(contour, inline=True, fontsize=8)
    plt.scatter(*vehicle_true_position, c='red', label='True Position')
    plt.scatter(landmarks[:, 0], landmarks[:, 1], marker='o', c='blue', label='Landmarks')
    plt.title(f'MAP Estimation Contours for K={K}')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
