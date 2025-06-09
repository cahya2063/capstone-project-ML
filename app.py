import gradio as gr
import numpy as np
from keras.preprocessing import image
from Model_Load import load_model_from_files
<<<<<<< HEAD
=======
from description import description

>>>>>>> 45c9f04 (Memperbaiki dan menyinkronkan HistoryLens)

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

<<<<<<< HEAD
    return f"{predicted_label} (Confidence: {confidence * 100:.2f}%)"
=======
    if confidence < 0.5:
        return "Tidak dapat dikenali", "Tolong arahkan ke objek yang jelas agar bisa diidentifikasikan. Pastikan anda berada di salah satu tempat Benteng_Vredeburg Candi_Borobudur, Candi_Prambanan, Gedung_Agung_Istana_Kepresidenan_Yogyakarta, Masjid_Gedhe_Kauman, Monumen_Serangan_1Maret, Museum_Gunungapi_Merapi, Situs_Ratu_Boko, Taman_Sari, Tugu_Yogyakarta"
    else:
        deskripsi = description.get(predicted_label, "Deskripsi belum tersedia.")
        label_output = f"{predicted_label} (Confidence: {confidence * 100:.2f}%)"
        return label_output, deskripsi



>>>>>>> 45c9f04 (Memperbaiki dan menyinkronkan HistoryLens)

# Buat antarmuka Gradio
demo = gr.Interface(
    fn=classify_image,
<<<<<<< HEAD
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Klasifikasi Cagar Budaya DIY",
    description="Upload gambar dan model akan mengklasifikasikannya ke dalam salah satu situs budaya di Yogyakarta."
)

=======
    inputs=gr.Image(type="pil", label="Upload Gambar"),
    outputs=[
        gr.Textbox(label="Output Klasifikasi"),
        gr.Textbox(label="Deskripsi Lengkap", lines=20, max_lines=50)
    ],
    allow_flagging="never",
    title="Klasifikasi Gambar",
    description="Upload gambar kami akan mengklasifikasikan dan memberikan deskripsi mengenai gambar tersebut."
)


>>>>>>> 45c9f04 (Memperbaiki dan menyinkronkan HistoryLens)
# Launch untuk Hugging Face Spaces
if __name__ == "__main__":
    demo.launch()