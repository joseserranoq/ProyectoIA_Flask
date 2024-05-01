
from flask import Flask, request
from flask_cors import CORS, cross_origin
import joblib
import json
import numpy as np

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


# correr pip -r requirements.txt para instalar las dependencias necesarias
# to run flask use: python -m flask run
# agreguen manualmente los archivos de pickle ya que son pesados, en este caso se genero una carpeta llamada pickle_files donde se agrego model8.pkl.

@app.route("/bitcoin", methods=['POST'])
@cross_origin()
def modelo1():
    '''
    {
    "Low": "3290.01"
    }
    '''
    data = request.json
    file = joblib.load('pickle_files/Modelo1_bitcoin.pkl')
    res = np.exp(file.predict([1, np.log(float(data['Low']))]))
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
            'pt': str(file[0, 0]),
            'pf': str(file[0, 1]),
            'fp': str(file[1, 0]),
            'fn': str(file[1, 1])
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
    res = file.predict([1, int(data['Total_Bags'])])
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
    res = np.exp(file.predict([1, np.log(int(data['Selling_Price']))]))
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
    res = file.predict([1, int(data['distance']), int(data['driver_tip'])])
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
    res = file.predict([1, int(data['open']), int(data['high']), int(data['low']), int(data['close'])])
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
    res = file.predict(
        [1, int(data['Age']), int(data['Weight']), int(data['Neck']), int(data['Chest']), int(data['Abdomen']),
         int(data['Hip']), int(data['Thigh']), int(data['Knee']), int(data['Ankle']), int(data['Biceps']),
         int(data['Forearm']), int(data['Wrist'])])
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
    res = file.predict([1, int(data['year']), int(data['month'])])
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
    res = file.predict([1, int(data['client_number']), int(data['open']), int(data['promo'])])
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
    file = joblib.load('pickle_files/model10.pkl')
    res = file.predict([1, int(data['department']), int(data['holiday']), int(data['month'])])
    return json.dumps({'result': res[0]})


@app.route('/emotion', methods=['POST'])
def get_emotion():
    """Detects faces in an image."""
    from google.cloud import vision
    import base64
    path = request.json
    client = vision.ImageAnnotatorClient()
    img_data = path['image'].encode()
    imageBase64 = base64.b64decode(img_data)

    with open("imageToSave.png", "wb") as fh:
        fh.write(imageBase64)

    with open('imageToSave.png', 'rb') as image_file:
        content = image_file.read()
    image = vision.Image(content=content)

    response = client.face_detection(image=image)
    faces = response.face_annotations

    # Names of likelihood from google.cloud.vision.enums
    likelihood_name = (
        "UNKNOWN",
        "VERY_UNLIKELY",
        "UNLIKELY",
        "POSSIBLE",
        "LIKELY",
        "VERY_LIKELY",
    )
    print("Faces:")

    for face in faces:
        print(f"anger: {likelihood_name[face.anger_likelihood]}")
        print(f"joy: {likelihood_name[face.joy_likelihood]}")
        print(f"surprise: {likelihood_name[face.surprise_likelihood]}")

        vertices = [
            f"({vertex.x},{vertex.y})" for vertex in face.bounding_poly.vertices
        ]

        print("face bounds: {}".format(",".join(vertices)))

    if response.error.message:
        raise Exception(
            "{}\nFor more info on error messages, check: "
            "https://cloud.google.com/apis/design/errors".format(response.error.message)
        )
    return json.dumps({
        'result': {
            'anger': likelihood_name[faces[0].anger_likelihood],
            'joy': likelihood_name[faces[0].joy_likelihood],
            'surprise': likelihood_name[faces[0].surprise_likelihood],

        }
    })
