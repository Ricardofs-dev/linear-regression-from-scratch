import numpy as np
import matplotlib.pyplot as plt
from linear_regression import LinearRegression

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic housing data
# X represents house size (e.g., in square meters)
X = np.random.rand(50, 1) * 100

# y represents house price
y = 2 * X.squeeze() + 10 + np.random.randn(50) * 10

# Create and train the linear regression model
model = LinearRegression(learning_rate=0.00001, n_iterations=1000)
model.fit(X, y)

# Plot the synthetic data
plt.figure()
plt.scatter(X, y)
plt.xlabel("House size")
plt.ylabel("House price")
plt.title("Synthetic housing data")

# Predict values using the trained model
y_pred = model.predict(X)

# Sort X for a clean regression line
idx = np.argsort(X[:, 0])
plt.plot(X[idx, 0], y_pred[idx], color="red")

plt.show(block=True)


