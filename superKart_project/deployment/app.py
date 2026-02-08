import pandas as pd
import joblib
from huggingface_hub import hf_hub_download
import gradio as gr


# -----------------------------
# Load model from HF Model Hub
# -----------------------------
def load_model():
    model_path = hf_hub_download(
        repo_id="MLbySush/super_kart_sales_model",
        filename="random_forest_model.joblib"
    )
    return joblib.load(model_path)

model = load_model()

# -----------------------------
# Prediction function
# -----------------------------
def predict_sales(
    item_weight,
    item_visibility,
    item_mrp
):
    data = pd.DataFrame([{
        "Item_Weight": item_weight,
        "Item_Visibility": item_visibility,
        "Item_MRP": item_mrp
    }])

    prediction = model.predict(data)[0]
    return round(float(prediction), 2)

# -----------------------------
# Gradio UI
# -----------------------------
interface = gr.Interface(
    fn=predict_sales,
    inputs=[
        gr.Number(label="Item Weight"),
        gr.Number(label="Item Visibility"),
        gr.Number(label="Item MRP")
    ],
    outputs=gr.Number(label="Predicted Sales"),
    title="SuperKart Sales Prediction",
    description="Predict product sales using a trained ML model hosted on Hugging Face."
)

interface.launch()

