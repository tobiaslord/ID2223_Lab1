import gradio as gr
from PIL import Image
import hopsworks

project = hopsworks.login()
fs = project.get_feature_store()

dataset_api = project.get_dataset_api()

dataset_api.download("Resources/images/latest_prediction.png")
dataset_api.download("Resources/images/latest_input.png")
dataset_api.download("Resources/images/df_recent.png")
dataset_api.download("Resources/images/confusion_matrix.png")

with gr.Blocks() as demo:
    with gr.Row():
      with gr.Column():
          gr.Label("Latest prediction")
          input_img = gr.Image("latest_prediction.png", elem_id="predicted-img")
      with gr.Column():
          gr.Label("Latest generated")
          input_img = gr.Image("latest_input.png", elem_id="actual-img")
    with gr.Row():
      with gr.Column():
          gr.Label("Recent Prediction History")
          input_img = gr.Image("df_recent.png", elem_id="recent-predictions")
      with gr.Column():
          gr.Label("Confusion Maxtrix with Historical Prediction Performance")
          input_img = gr.Image("confusion_matrix.png", elem_id="confusion-matrix")

demo.launch()
