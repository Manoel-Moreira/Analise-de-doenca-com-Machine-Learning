# 1 - chamar o framework que tem o TF-IDF
# 2 - instanciar o modelo de vetorização
# 3 - Pegar os dados e vetorizar

from sklearn.feature_extraction.text import TfidfVectorizer
from services.datasetService import dataset_completo
from sklearn.preprocessing import LabelEncoder


# Vetorizador
def vetorizador():
    """
    Funcao que cria um vetorizador TF-IDF e o ajusta aos dados de entrada.

    Returns:
        Vetorizador ajustado com os dados pronto para ser usado.
    """
    tfidf = TfidfVectorizer()

    df = dataset_completo()
    
    X = df["sintomas"].astype(str)

    tfidf.fit(X)
    
    return tfidf

# Vetorizador vetorizando os Dados
# Retona os Dados Vetorizados
def vetorizacao():
    tfidf = TfidfVectorizer()

    df = dataset_completo()
    X = df["sintomas"].astype(str)

    tfidf.fit(X)

    X_tfidf = tfidf.transform(X)

    return X_tfidf

def encode_Y():
    """
    Função para codificar a variável alvo (y) usando Label Encoding.

    Return:
    Y_encoded: array
        Array com os rótulos codificados.
    """
    label_encoder = LabelEncoder()
    df = dataset_completo()

    y = df["diagnostico"].astype(str)

    y_encoded = label_encoder.fit_transform(y)

    return y_encoded










