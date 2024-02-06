from flask import Flask, request, jsonify
from flask_firebase_admin import FirebaseAdmin
from PIL import Image
import numpy as np
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from flask_cors import CORS
from google.cloud import storage
from tensorflow import lite
import os

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = 'microservice-413007-dba7d34bbce5.json'

app = Flask(__name__)
firebase = FirebaseAdmin(app)
CORS(app)

# Inisialisasi Cloud Storage
storage_client = storage.Client()
bucket = storage_client.bucket('crab_bucket')

# Load existing TFLite model
model_tflite = lite.Interpreter(model_path="model\model_v2.tflite")
model_tflite.allocate_tensors()

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'tiff', 'webp', 'jfif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

confidence_threshold = 0.8

# Variabel penyimpanan sementara untuk data yang di-POST
posted_data = []

@app.route('/klasifikasi', methods=['POST'])
def predict():
    try:
        # Get the uploaded image file
        image_file = request.files['image']

        if image_file and allowed_file(image_file.filename):
            img = Image.open(image_file).convert('RGB')
            img = img.resize((300, 300))
            x = image.img_to_array(img)
            x = x / 255.0
            images = np.expand_dims(x, axis=0)

            # Data augmentation
            datagen = ImageDataGenerator(
                rotation_range=40,
                shear_range=0.2,
                width_shift_range=0.2,
                height_shift_range=0.2,
                horizontal_flip=True,
                fill_mode='nearest'
            )

            # Generate augmented images
            augmented_images = datagen.flow(images)
            inputs = next(augmented_images)

            # Perform predictions on the augmented images using TFLite model
            # Get input and output tensors
            input_tensor_index = model_tflite.get_input_details()[0]['index']
            output = model_tflite.get_tensor(model_tflite.get_output_details()[0]['index'])

            model_tflite.set_tensor(input_tensor_index, inputs)
            model_tflite.invoke()
            prediction_array_tflite = output

            average_prediction = np.mean(prediction_array_tflite, axis=0)

            class_names = ['Kepiting Biasa', 'Kepiting Soka']

            # Check confidence level
            confidence_tflite = np.max(average_prediction)
            if confidence_tflite < confidence_threshold:
                return jsonify({"error": "Kepiting tidak terdeteksi dengan keyakinan yang cukup."}), 400

            # Format the response JSON
            predictions = {
                "prediction_tflite": class_names[np.argmax(average_prediction)],
                "confidence_tflite": '{:2.0f}%'.format(100 * np.max(average_prediction)),
            }

            # Simpan data yang di-POST ke variabel global
            posted_data.append(predictions)

            # Set file pointer to the beginning
            image_file.seek(0)
            # Simpan file di Cloud Storage
            blob = bucket.blob('images/' + image_file.filename)
            # Upload image from byte string
            blob.upload_from_string(image_file.read(), content_type='image/jpeg')

            return jsonify(predictions)

        else:
            return jsonify({"error": "Invalid file format."}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Endpoint untuk metode GET
@app.route('/get_data', methods=['GET'])
def get_posted_data():
    global posted_data
    if posted_data:
        return jsonify({"posted_data": posted_data})
    else:
        return jsonify({"message": "No data has been posted yet."})

# Endpoint untuk metode GET baru
@app.route('/crabify', methods=['GET'])
def hello():
    return jsonify({"message": "Hello, this is a GET request!"})

# Jalankan aplikasi Flask
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
