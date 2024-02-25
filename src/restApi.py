from flask import Flask, request, jsonify
import tensorflow as tf

filepath = "src/resources/zeroShotImage.model"

app = Flask(__name__)
loaded_model = tf.saved_model.load(filepath)

@app.route('/predict', methods=['POST'])
def predict():
    # Assume input image data is in request.json
    input_data = request.json
    # Make predictions using the loaded model
    predictions = loaded_model(input_data)
    return jsonify(predictions)

if __name__ == '__main__':
    app.run(debug=True)
