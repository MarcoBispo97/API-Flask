from flask import Flask
from textblob import TextBlob
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


df = pd.read_csv(
    'C:/Users/marco/OneDrive/Dataside/Projeto_ML_Titanic/meuprojeto/mlops-machine-learning-aula-3/casas.csv')
colunas = ['tamanho','preco']
df = df[colunas]
X = df.drop('preco', axis=1)
y = df['preco']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)
modelo = LinearRegression()
modelo.fit(X_train.values, y_train)

app = Flask(__name__)
#Definindo as rotas das API:
#O que vai acontecer quando alguem chegar aqui
@app.route('/')
def home():
    return "Minha primeira API."

@app.route('/cotacao/<int:tamanho>')
def cotacao(tamanho):
    preco = modelo.predict([[tamanho]])
    return str(preco)


app.run(debug=True)

#debug=True Mudança automática de debug
#O processamento do modelo acontece dentro do servidor do Flask