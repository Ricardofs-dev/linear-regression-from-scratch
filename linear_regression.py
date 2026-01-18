import numpy as np


class LinearRegression:
    """
    Regressão Linear utilizando Gradiente Descendente.

    Esta classe permite:
    - Ajustar um modelo linear aos dados
    - Prever novos valores com base no modelo treinado
    """

    def __init__(self, learning_rate=0.01, n_iterations=1000):
        """
        Inicializa os parâmetros do modelo.

        Parâmetros:
        - learning_rate: taxa de aprendizagem do gradiente descendente
        - n_iterations: número máximo de iterações de treino
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations

        # Parâmetros do modelo (a serem aprendidos durante o treino)
        self.weight = None
        self.bias = None

    def fit(self, X, y, min_error=None, min_delta=None):
        """
        Treina o modelo de regressão linear utilizando gradiente descendente.

        Passos do algoritmo:
        1. Inicializar o peso e o bias
        2. Iterar para um número fixo de iterações ou até atingir o critério de paragem
        3. Calcular as previsões
        4. Calcular o erro
        5. Atualizar o peso e o bias
        """
        # Número de amostras de treino
        n_samples = X.shape[0]

        # Inicialização dos parâmetros
        self.weight = 0.0
        self.bias = 0.0

        # Erro da iteração anterior (para paragem por variação mínima)
        previous_error = float("inf")
        
        # Gradient Descent loop
        for _ in range(self.n_iterations):
            # Previsões do modelo
            y_pred = self.weight * X + self.bias
            errors = y_pred - y

            # Erro quadrático médio (MSE)
            mse = np.mean(errors ** 2)

            # Condição de paragem por erro mínimo
            if min_error is not None and mse <= min_error:
                break
            # Condição de paragem por variação mínima do erro
            if min_delta is not None and abs(previous_error - mse) < min_delta:
                break
            
            previous_error = mse

            
            # Cálculo dos gradientes
            dw = (1 / n_samples) * np.sum(errors * X)
            db = (1 / n_samples) * np.sum(errors)

            # Atualização dos parâmetros
            self.weight -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            pass


    def predict(self, X):
       """
       Prevê os valores de saída para os dados de entrada X.
       """
       return self.weight * X + self.bias

