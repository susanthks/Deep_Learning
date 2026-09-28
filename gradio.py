import gradio as gr

def predict_image(image):

    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)

    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    prediction = model.predict(image_array, verbose=0)[0]

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
    title="🌍 Scene Image Classification",
    description=(
        "Upload a real-world image and the trained "
        f"{winner} model will classify it into one of six categories."
    )
)

demo.launch(share=True)
