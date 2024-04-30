from flask import Flask,request
import joblib
app = Flask(__name__)

#correr pip -r requirements.txt para instalar las dependencias necesarias
#to run flask use: python -m flask run

@app.route("/", methods=['GET', 'POST'])
def home():
    data = request.json
    #agreguen manualmente los archivos de pickle ya que son pesados, en este caso se genero una carpeta llamada pickle_files donde se agrego model8.pkl.
    file = joblib.load('pickle_files/model8.pkl')
    print()
    res = file.predict([1,int(data['param1']),int(data['param2'])])
    return f"{res[0]}"

