import json
import numpy as np
import gradio as gr
from PIL import Image
from tensorflow import keras


IMG_SIZE = (128, 128)

model = keras.models.load_model(
    "models/best_model.keras"
)

with open("models/class_names.json", "r") as f:
    class_names = json.load(f)


def predict_image(image):

    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)

    image_array = np.array(image) / 255.0

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    prediction = model.predict(
        image_array,
        verbose=0
    )[0]

    results = {
        class_names[i]: float(prediction[i])
        for i in range(len(class_names))
    }

    return results


demo = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(
        type="pil",
        label="Upload Image"
    ),
    outputs=gr.Label(
        num_top_classes=6,
        label="Prediction"
    ),
    title="Scene Image Classification",
    description=(
        "Upload an image and the trained model "
        "will predict the scene category."
    )
)


if __name__ == "__main__":
    demo.launch()
