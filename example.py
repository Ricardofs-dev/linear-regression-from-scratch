import numpy as np
import matplotlib.pyplot as plt
from linear_regression import LinearRegression

# Definir seed aleatória
np.random.seed(42)

# Gerar dados sintéticos de habitação
# X representa o tamanho da casa (por exemplo, em metros quadrados)
X = np.random.rand(50, 1) * 100

# y representa o preço da casa (por exemplo, em milhares de euros)
y = 2 * X.squeeze() + 10 + np.random.randn(50) * 10

# # Criar e treinar o modelo de regressão linear
model = LinearRegression(learning_rate=0.00001, n_iterations=1000)
model.fit(X, y, min_error=1.0, min_delta=0.0001)

# Plot dos dados e da linha de regressão
plt.figure()
plt.scatter(X, y)
plt.xlabel("House size")
plt.ylabel("House price")
plt.title("Synthetic housing data")

# Prever valores utilizando o modelo treinado
y_pred = model.predict(X)

# Ordenar X para desenhar uma linha de regressão contínua
idx = np.argsort(X[:, 0])
plt.plot(X[idx, 0], y_pred[idx], color="red")

plt.show(block=True)