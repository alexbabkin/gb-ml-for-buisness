from flask_cors import CORS, cross_origin
from time import strftime
import flask
import dill
import pandas as pd
import numpy as np
import os
import json
dill._dill._reverse_typemap['ClassType'] = type

# initialize our Flask application and the model
app = flask.Flask(__name__)
CORS(app)
model = None


def load_model(model_path):
    global model
    with open(model_path, 'rb') as f:
        model = dill.load(f)
    print(model)


modelpath = os.environ.get('MODELS_PATH', "models/model.dill")
load_model(modelpath)


@app.route("/", methods=["GET"])
def general():
    return """Welcome to fraudelent prediction process. Please use 'http://<address>/predict' to POST"""


@app.route("/predict", methods=["POST"])
@cross_origin()
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": False}
    dt = strftime("[%Y-%b-%d %H:%M:%S]")
    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        request_json = json.loads(flask.request.get_json())

        embarked = request_json['Embarked']
        sex = request_json['Sex']
        pclass = request_json['Pclass']
        age = request_json['Age']
        sibsp = request_json['SibSp']
        parch = request_json['Parch']
        fare = request_json['Fare']
        cabin = request_json['Cabin']

        try:
            preds = model.predict(
                pd.DataFrame({
                    "Embarked": [embarked or np.nan],
                    "Sex": [sex or np.nan],
                    "Pclass": [pclass or np.nan],
                    "Age": [age or np.nan],
                    "SibSp": [sibsp or np.nan],
                    "Parch": [parch or np.nan],
                    "Fare": [fare or np.nan],
                    "Cabin": [cabin or np.nan]
                })
            )



        except AttributeError as e:
            data['predictions'] = str(e)
            data['success'] = False
            return flask.jsonify(data)

        data["predictions"] = str(preds[0])

        data["success"] = True

    return flask.jsonify(data)


# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    print("* Loading the model and Flask starting server...")
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', debug=True, port=port)
