# Regressão Linear do Zero

Este projeto implementa um modelo de **Regressão Linear do zero**, utilizando apenas a biblioteca `numpy`.
O objetivo é demonstrar uma implementação clara e educativa do algoritmo, sem recorrer a bibliotecas
de Machine Learning que encapsulem o processo (como scikit-learn).

Este trabalho foi desenvolvido no âmbito da disciplina de **Programação** da
Pós-Graduação em **Data Science and Business Intelligence**.

---

## 📌 Descrição do Projeto

A Regressão Linear é um algoritmo de aprendizagem supervisionada que procura modelar
a relação entre uma variável de entrada e uma variável de saída contínua, ajustando
uma função linear aos dados.

Neste projeto:
- O modelo é treinado utilizando **Gradiente Descendente**
- Os parâmetros `weight` (inclinação) e `bias` (interceção) são aprendidos iterativamente
- A implementação privilegia a **clareza e legibilidade do código**, em detrimento da performance

---

## ⚙️ Descrição do Algoritmo

O processo de treino do modelo segue os seguintes passos:

1. Inicialização dos parâmetros (`weight` e `bias`)
2. Cálculo das previsões com base no modelo linear
3. Cálculo do erro entre os valores previstos e os valores reais
4. Atualização dos parâmetros utilizando gradiente descendente
5. Repetição do processo durante um número fixo de iterações

Como a função de custo da regressão linear é convexa, o algoritmo converge para um
**ótimo global**.

---

## ▶️ Como Executar o Projeto

1. Clonar o repositório:
```bash
git clone <url-do-repositorio>

pip install -r requirements.txt

python example.py



