from flask import Flask, render_template, request
import os
import numpy as np
from keras.preprocessing import image
from Model_Load import load_model_from_files

# Inisialisasi Flask
app = Flask(__name__)

# Load model
model = load_model_from_files("model.json", "my_model.h5")

# Label prediksi sesuai urutan output model
labels = [
    "Benteng Vredeburg", "Candi Borobudur", "Candi Prambanan", "Gedung Agung Istana Kepresidenan",
    "Masjid Gedhe Kauman", "Monumen Serangan 1 Maret", "Museum Gunungapi Merapi",
    "Situs Ratu Boko", "Taman Sari", "Tugu Yogyakarta"
]

# untuk menyimpan file upload sementara
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'static')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Fungsi preprocessing gambar
def prepare_image(filepath):
    img = image.load_img(filepath, target_size=(224, 224, 3))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array / 255.0



@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    img_url = None  # opsional

    if request.method == "POST":
        if 'image' not in request.files:
            return render_template("input.html", prediction="Tidak ada file yang diupload!", img_path=None)
        
        file = request.files["image"]
        if file.filename == '':
            return render_template("input.html", prediction="File tidak valid!", img_path=None)
        
        # Buat nama file unik
        import uuid
        filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]

        # Simpan ke folder static
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        # Processing gambar
        processed_img = prepare_image(filepath)
        pred = model.predict(processed_img)[0]
        confidence = np.max(pred)
        predicted_label = labels[np.argmax(pred)]
        prediction = f"Gambar diklasifikasikan sebagai: {predicted_label} (Confidence: {confidence * 100:.2f}%)"

        img_url = f"/static/{filename}"  # URL (Opsional)

    return render_template("input.html", prediction=prediction, img_path=img_url)



if __name__ == "__main__":
    app.run(debug=True)


