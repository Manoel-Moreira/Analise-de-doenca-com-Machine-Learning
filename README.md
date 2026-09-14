# Análise de Doenças com Machine Learning

Projeto desenvolvido através da Mastertech do professor Carlos Viana para colocar em prática conceitos básicos de processamento de texto e classificação utilizando Machine Learning. O objetivo foi criar um modelo capaz de analisar dados de diagnósticos médicos e prever condições com base nas informações fornecidas.

---

## O que foi feito no projeto

1. **Dados Prontos (ETL):** Os dados utilizados para o treinamento já passaram pela fase de limpeza e preparação prévia.
2. **Vetorização dos Dados (TF-IDF):** 
   - Como os modelos matemáticos não entendem texto puro, o primeiro passo foi transformar os dados textuais em números.
   - Para isso, utilizei o `TfidfVectorizer` da biblioteca `scikit-learn`. Ele ajuda a medir a importância de cada palavra no conjunto de dados.
3. **Treinamento do Modelo (XGBoost):**
   - Com os dados convertidos em vetores, treinei o modelo de classificação usando o `XGBClassifier` da biblioteca `XGBoost`.
   - Escolhi o XGBoost por ser um algoritmo muito eficiente e amplamente utilizado em problemas de classificação.

---

##  Tecnologias e Bibliotecas Utilizadas

- **Python** 
- **scikit-learn** (Para vetorização de texto com `TfidfVectorizer`)
- **XGBoost** (Para criação e treinamento do modelo com `XGBClassifier`
- **FastApi** (Para a interface)
  

---

## 🚀 Como executar o projeto

1. Clone o repositório:
   ```bash
   git clone [https://github.com/Manoel-Moreira/Masterclass-analise-de-doenca-com-Machine-Learning.git](https://github.com/Manoel-Moreira/Masterclass-analise-de-doenca-com-Machine-Learning.git)
