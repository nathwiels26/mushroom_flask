from flask import Flask, render_template, request, jsonify
from flask_cors import CORS   # Jika akses dari luar port
import tensorflow as tf
import numpy as np

app = Flask(__name__)
CORS(app)  # Optional, hanya jika akses dari luar port
model = tf.keras.models.load_model('mushroom_model.h5')

selected_features = [
    'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
    'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
    'stalk-shape', 'stalk-root', 'ring-type'
]

@app.route('/', methods=['GET'])
def home():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if data is None:
        return jsonify({'error': 'Data kosong'})
    try:
        x = np.array([[data[f] for f in selected_features]], dtype=np.float32)
        prob = float(model.predict(x)[0][0])
        pred = int(prob >= 0.5)
        return jsonify({'probability': prob, 'prediction': pred})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False)
