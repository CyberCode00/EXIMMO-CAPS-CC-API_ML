from flask import Flask, request, jsonify
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import numpy as np
from flask_cors import CORS
from urllib.parse import quote

app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    # Input needs
    data = request.get_json()

    # Check if all required fields are present
    if 'commodity' not in data or 'region' not in data or 'year' not in data:
        return jsonify({'error': 'Missing input values'}), 400

    mode = data.get('mode')
    commodity = data['commodity']
    region = data['region']
    starting_year = int(data['year'])  # Starting year for prediction
    num_years_to_predict = 4  # Adjust the number of years to predict as needed

    # Encode special characters in the model name
    encoded_commodity = quote(commodity)

    # Construct the model path with the encoded commodity name
    model_path = f"F:\API Flask\data\Predict model\{mode}\H5 {encoded_commodity} {mode}\{encoded_commodity}({region}).h5"

    # Add try-except block to catch file not found errors
    try:
        model = tf.keras.models.load_model(model_path)
    except FileNotFoundError:
        return jsonify({'error': f'Model file not found: {model_path}'}), 404

    # Get the expected sequence length from the model
    sequence_length = model.layers[0].input_shape[1]

    scaler = MinMaxScaler(feature_range=(0, 1))

    # Initialize a list to store predicted prices
    predicted_prices = {}

    for i in range(num_years_to_predict):
    # Increment the year for each iteration
        year = starting_year + i

        # Add some variation to the input data
        random_variation = np.random.uniform(low=-0.1, high=0.1, size=(1, sequence_length, 1))
        years = np.array([[year]])
        years = np.tile(years, (1, sequence_length, 1))
        year_scaled = scaler.fit_transform((years + random_variation).reshape(-1, 1)).reshape(1, sequence_length, 1)

        # Ensure the batch dimension is included
        predicted_price = model.predict(year_scaled)

        # Convert the NumPy float32 to Python float
        predicted_price = float(predicted_price[0, 0])

        # Add the predicted price to the dictionary
        predicted_prices[f'predicted price on {year}'] = predicted_price

    # Convert the list of predicted prices to a JSON response
    return jsonify({'predicted_prices': predicted_prices})

if __name__ == '__main__':
    app.run(port=3000, debug=True)
