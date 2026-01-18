# Regressão Linear  – Previsão de Preços das Casas

Este projeto implementa um modelo de **Regressão Linear**, utilizando apenas a biblioteca `numpy`.
O objetivo é demonstrar uma implementação clara e educativa do algoritmo, sem recorrer a bibliotecas
de Machine Learning (como scikit-learn).

O caso de estudo utilizado consiste em **prever o preço de casas com base no tamanho** (por exemplo, área em m²), recorrendo a **dados sintéticos gerados aleatoriamente** com uma relação aproximadamente linear entre as variáveis.

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
- A implementação privilegia a **clareza e legibilidade do código**, em detrimento da performance.

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

## Condições de Paragem

O treino do modelo pode ser interrompido através de:

- Número máximo de iterações (`n_iterations`)
- Erro mínimo (`min_error`)
- Variação mínima do erro entre iterações (`min_delta`)

Estas condições permitem um controlo mais flexível do processo de treino,
evitando iterações desnecessárias.

---

## ▶️ Como Executar o Projeto

1. Clonar o repositório:
```bash
git clone <url-do-repositorio>
pip install -r requirements.txt
python example.py

Este projeto utiliza apenas bibliotecas básicas (`numpy` e `matplotlib`), não recorrendo a algoritmos de Machine Learning pré-implementados.



