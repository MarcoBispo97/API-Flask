from flask import Flask
from textblob import TextBlob

app = Flask(__name__)

#Definindo as rotas das API:
#O que vai acontecer quando alguem chegar aqui
@app.route('/')
def home():
    return "Minha primeira API."

@app.route('/sentimento/<frase>')
def sentimeto(frase):
    tb = TextBlob(frase)
    tb_en = tb.translate(from_lang='pt_br', to='en')
    polaridade = tb_en.sentiment.polarity
    return "polaridade: {}".format(polaridade)


app.run(debug=True)

#debug=True Mudança automática de debug

#O processamento do modelo acontece dentro do servidor do Flask