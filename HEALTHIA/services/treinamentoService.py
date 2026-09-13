"""
 --- Passo a passo de como será feito o treinamento --

#Passo 1 -
# Buscar os dados - X, Y

#Passo 2 -
# Separar os dados em treino (75%) (X_treino, Y_treino) e teste (25%) (X_teste, Y_teste)


#Passo 3 -
# Treinar o modelo com os dados de treino


#Passo 4 -
# Avaliar o modelo com os dados de teste (Acurácia)

"""
from services.vetorizacaoService import vetorizacao, encode_Y
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier as XGBoost #xgboost --> um algoritmo de Aprendizado de Máquina supervisionado escolhido para fazer o treinamento


def buscar_dados():
    # Implementar a lógica para buscar os dados - X, Y
    X = vetorizacao()
    y = encode_Y()
    return X, y

def separar_dados():
    
    #buscando os dadps
    X, y = buscar_dados()

    # Separar os dados em treino (70%) e teste (30%)
    X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.3, random_state=42)
    return X_treino, X_teste, y_treino, y_teste

def treinar_modelo():
    """
    Função para treinar o modelo de Machine Learning, com o algoritmo XGBoost.

    Return:
    model: XGBClassifier
        O modelo treinado - que se chama HealthIA.
    """
    # Implementar a lógica para treinar o modelo

    X_treino = separar_dados()[0] #--> Pega somente o X_treino da função
    y_treino = separar_dados()[2]  #--> Pega somente o y_treino da função

    #deixando o modelo mais ponte, com bons hiperparametros
    HealthIA = XGBoost(n_estimators=150, learning_rate=0.03, max_depth=3, min_child_weight=1, random_state=42) # --> Cria o modelo - entre parenteses são os hiperparâmetros.
    HealthIA.fit(X_treino, y_treino) # --> Treina o modelo

    return HealthIA

def acuracia_modelo():

    HealthIA = treinar_modelo()
    X_teste = separar_dados()[1] #--> Pega somente o X_teste da função
    y_teste = separar_dados()[3] #--> Pega somente o y_teste da função

   
    y_pred = HealthIA.predict(X_teste)
    acuracia = accuracy_score(y_teste, y_pred) #--> Pega a resposta correta com o que o modelo previu.

    porcentagem = acuracia * 100

    return porcentagem 
