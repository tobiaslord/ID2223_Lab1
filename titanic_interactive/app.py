import gradio as gr
import numpy as np
from PIL import Image
import requests

import hopsworks
import joblib
import pandas as pd

project = hopsworks.login()
fs = project.get_feature_store()


mr = project.get_model_registry()
model = mr.get_model("titanic", version=1)
model_dir = model.download()
model = joblib.load(model_dir + "/titanic_model.pkl")

CLASS_TO_VALUE = {
    "1st class": "1",
    "2nd class": "2",
    "3rd class": "3",
}

PORT_TO_VALUE = {
    "Cherbourg": "C",
    "Queenstown": "Q",
    "Southampton": "S",
}


def titanic(ticket_class, sex, port, fare, age, sibsp, parch):
    data = {
        "pclass": [CLASS_TO_VALUE[ticket_class]],
        "sex": [sex],
        "embarked": [PORT_TO_VALUE[port]],
        "fare": [fare],
        "age": [age],
        "sibsp": [int(sibsp)],
        "parch": [int(parch)],
    }
    df = pd.DataFrame(data)
    # 'res' is a list of predictions returned as the label.
    res = model.predict(df)
    # We add '[0]' to the result of the transformed 'res', because 'res' is a list, and we only want
    # the first element.

    if res:
        url = "https://m.media-amazon.com/images/I/71M6k7ZQNcL._RI_.jpg"
    else:
        url = "https://thumbs.dreamstime.com/b/allvarlig-sten-med-skallen-34707626.jpg"

    img = Image.open(requests.get(url, stream=True).raw)
    return img

demo = gr.Interface(
    fn=titanic,
    title="Titanic survival prediction",
    description="Experiment with parameters to predict if the fictional passenger survived",
    allow_flagging="never",
    inputs=[
        gr.inputs.Dropdown(["1st class", "2nd class", "3rd class"], label="Ticket class"),
        gr.inputs.Dropdown(["female", "male"], label="Sex"),
        gr.inputs.Dropdown(["Cherbourg", "Queenstown", "Southampton"], label="Port of Embarkation"),
        gr.inputs.Number(default=50.0, label="Fare"),
        gr.inputs.Number(default=20.0, label="Age"),
        gr.inputs.Number(default=0, label="Number of siblings/spouses aboard the Titanic"),
        gr.inputs.Number(default=0, label="Number of parents/children aboard the Titanic"),
        ],
    outputs=gr.Image(type="pil"))

demo.launch()
