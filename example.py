import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic housing data
# X represents house size (e.g., in square meters)
X = np.random.rand(50, 1) * 100

# y represents house price
y = 2 * X.squeeze() + 10 + np.random.randn(50) * 10

# Plot the synthetic data
plt.scatter(X, y)
plt.xlabel("House size")
plt.ylabel("House price")
plt.title("Synthetic housing data")
plt.show()
