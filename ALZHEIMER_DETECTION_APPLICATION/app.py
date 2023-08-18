from flask import Flask, render_template, request
import keras.models
import numpy as np
from keras.utils import load_img
import firebase_admin
from firebase_admin import credentials, storage
from werkzeug.exceptions import BadRequestKeyError

app = Flask(__name__)

# Initialize Firebase Admin SDK with your private key
cred = credentials.Certificate("static/alzheimer--sdiseases-firebase-adminsdk-zt35t-df6848472a.json")
firebase_admin.initialize_app(cred, {'storageBucket': 'alzheimer--sdiseases.appspot.com'})

model = keras.models.load_model(r"finalmodel.h5")

label_to_class = {'Mild Demented': 0,
                  'Moderate Demented': 1,
                  'Non Demented': 2,
                  'Very Mild Demented': 3}


def predict(img_path):
    image = load_img(img_path, target_size=(224, 224))
    image = np.expand_dims(image, axis=0)
    preds = model.predict(image)
    i = np.argmax(preds[-1])
    class_to_label = {v: k for k, v in label_to_class.items()}
    label = class_to_label[i]
    return label


@app.route('/', methods=['GET', 'POST'])
def main():
    return render_template("home.html")


@app.route('/mild')
def mild():
    return render_template("mild.html")


@app.route('/moderate')
def moderate():
    return render_template("moderate.html")


@app.route('/verymild')
def verymild():
    return render_template("verymild.html")


@app.route('/scan')
def scan():
    return render_template("scan.html")


@app.route("/submit", methods=['GET', 'POST'])
def get_output():
    try:
        img = request.files["my_image"]
        img_filename = img.filename
        # process the image and get the prediction
    except BadRequestKeyError:
        error_message = "Please select an image to upload."
        return render_template("index.html", error_message=error_message)

    if request.method == 'POST':
        img = request.files['my_image']

        img_path = "static/" + img_filename
        img.save(img_path)

        # Upload the image to Firebase Storage
        with open(img_path, "rb") as f:
            image_data = f.read()

        bucket = storage.bucket()
        blob = bucket.blob("images/" + img_filename)
        blob.upload_from_string(image_data, content_type="image/jpeg")

        p = predict(img_path)

    return render_template("scan.html", prediction=p, img_path=img_path)


if __name__ == '__main__':
    # app.debug = True
    app.run(debug=True)
