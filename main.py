# importing libraries
import os
import pickle
import requests

from flask import Flask, render_template, request

from fastai.vision import load_learner, Path, open_image

# creating a fastai path object for seamless navigation
path = Path()

# creating model object from pickle file
url = 'https://drive.google.com/uc?export=download&confirm=_js9&id=10bHBHw5-prtrZd6R_PgiLfDHRZ2CBn0z'
r = requests.get(url, allow_redirects=True)
open('model/model.pkl', 'wb').write(r.content)

model = load_learner(path, 'model/model.pkl')

# creating flask app
app = Flask(__name__)

upload_folder = "static/" # user uploaded image will be saved here

@app.route("/", methods=["GET", "POST"])

def upload_predict():
    if request.method == "POST":
        image_file = request.files["image"]
        if image_file:
            image_location = os.path.join(
                upload_folder,
                image_file.filename
            )
            image_file.save(image_location)
            img = open_image(image_location)
            prediction, _, _ = model.predict(img)

            return render_template("predict.html", output=prediction.obj)
    return render_template("index.html")

# for running natively on user device
if __name__ == "__main__":
    app.run(port=5000, debug=True)
