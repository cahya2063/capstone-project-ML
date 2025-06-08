import gradio as gr
import numpy as np
from keras.preprocessing import image
from Model_Load import load_model_from_files

# Load model dan label
model = load_model_from_files("model.json", "my_model.h5")

labels = [
    "Benteng Vredeburg", "Candi Borobudur", "Candi Prambanan", "Gedung Agung Istana Kepresidenan",
    "Masjid Gedhe Kauman", "Monumen Serangan 1 Maret", "Museum Gunungapi Merapi",
    "Situs Ratu Boko", "Taman Sari", "Tugu Yogyakarta"
]

# Fungsi preprocessing dan prediksi
def classify_image(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    pred = model.predict(img_array)[0]
    confidence = np.max(pred)
    predicted_label = labels[np.argmax(pred)]

    return f"{predicted_label} (Confidence: {confidence * 100:.2f}%)"

# Buat antarmuka Gradio
demo = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Klasifikasi Cagar Budaya DIY",
    description="Upload gambar dan model akan mengklasifikasikannya ke dalam salah satu situs budaya di Yogyakarta."
)

# Launch untuk Hugging Face Spaces
if __name__ == "__main__":
    demo.launch()