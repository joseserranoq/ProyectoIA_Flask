from flask import Flask,request
import joblib
import json
import numpy as np
app = Flask(__name__)

#correr pip -r requirements.txt para instalar las dependencias necesarias
#to run flask use: python -m flask run
    #agreguen manualmente los archivos de pickle ya que son pesados, en este caso se genero una carpeta llamada pickle_files donde se agrego model8.pkl.

@app.route("/bitcoin", methods=['POST'])
def modelo1():
    '''
    {
    "Low": "3290.01"
    }
    '''
    data = request.json
    file = joblib.load('pickle_files/Modelo1_bitcoin.pkl')
    res = np.exp(file.predict([1,np.log(float(data['Low']))]))
    return json.dumps({'result': res[0]})

@app.route("/wine", methods=['GET'])
def modelo2():
    '''
    [[599 235]
    [219 432]]
    '''
    file = joblib.load('pickle_files/Modelo_2.pkl')
    print(file)
    data = json.dumps(
        {
            'pt': str(file[0,0]),
            'pf': str(file[0,1]),
            'fp': str(file[1,0]),
            'fn': str(file[1,1])
        })
    return data

@app.route("/avocado", methods=['POST'])
def modelo3():
    
    '''
    {
    "Total_Bags": "1000"
    }
    '''

    data = request.json
    file = joblib.load('pickle_files/Modelo_3_Aguacate.pkl')
    res = file.predict([1,int(data['Total_Bags'])])
    return json.dumps({'result': res[0]})

@app.route("/car_price", methods=['POST'])
def modelo4():
    
    '''
    {
    "Selling_Price": "10"
    }
    '''

    data = request.json
    file = joblib.load('pickle_files/Modelo_4_Car.pkl')
    res = np.exp(file.predict([1,np.log(int(data['Selling_Price']))]))
    return json.dumps({'result': res[0]})

@app.route("/bicycle", methods=['POST'])
def modelo5():
    
    '''
    {
    "distance": "52", 
    "driver_tip": "3000"
    }
    '''
    data = request.json
    file = joblib.load('pickle_files/Modelo_5_Trip.pkl')
    res = file.predict([1,int(data['distance']),int(data['driver_tip'])])
    return json.dumps({'result': res[0]})



@app.route("/sp500stock", methods=['POST'])
def modelo6():

    '''
    {
    "open": "1",
    "high": "1",
    "low": "1",
    "close": "1"
    }
    '''
    data = request.json
    file = joblib.load('pickle_files/model6.pkl')
    res = file.predict([1,int(data['open']),int(data['high']),int(data['low']),int(data['close'])])
    return json.dumps({'result': res[0]})

@app.route("/bodymass", methods=['POST'])
def modelo7():
    
    '''
    {
        "Age": "23",
        "Weight": "154",
        "Neck": "36",
        "Chest": "93",
        "Abdomen": "85",
        "Hip": "94",
        "Thigh": "59",
        "Knee": "37",
        "Ankle": "21",
        "Biceps": "27",
        "Forearm": "27",
        "Wrist": "17"
    }
    '''
    data = request.json
    file = joblib.load('pickle_files/model7.pkl')
    res = file.predict([1,int(data['Age']),int(data['Weight']),int(data['Neck']),int(data['Chest']),int(data['Abdomen']),int(data['Hip']),int(data['Thigh']),int(data['Knee']),int(data['Ankle']),int(data['Biceps']),int(data['Forearm']),int(data['Wrist'])])
    return json.dumps({'result': res[0]})

@app.route("/company", methods=['POST'])
def modelo8():
    
    '''
    {
        "year": "2013",
        "month": "1"
    }
    '''

    data = request.json
    file = joblib.load('pickle_files/model8.pkl')
    res = file.predict([1,int(data['year']),int(data['month'])])
    return json.dumps({'result': res[0]})

@app.route("/rossman", methods=['POST'])
def modelo9():
    '''
    {
        "client_number": "625",
        "open": "1",
        "promo": "1"
    }
    '''
    data = request.json
    file = joblib.load('pickle_files/model9.pkl')
    res = file.predict([1,int(data['client_number']),int(data['open']),int(data['promo'])])
    return json.dumps({'result': res[0]})

@app.route("/walmart", methods=['POST'])
def modelo10():
    '''
    {
        "department": "1",
        "holiday": "1",
        "month": "2"
    }
    '''
    data = request.json
    file = joblib.load('pickle_files/model7.pkl')
    res = file.predict([1,int(data['param1']),int(data['param2'])])
    return json.dumps({'result': res[0]})