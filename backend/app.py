# from flask import Flask, jsonify
# from flask_cors import CORS
# import yfinance as yf
# import numpy as np
# import pickle
# from tensorflow.keras.models import load_model

# app = Flask(__name__)
# CORS(app)

# # Load your model & scaler
# model = load_model("eth_price_model.h5")
# with open("scaler.pkl", "rb") as f:
#     scaler = pickle.load(f)

# @app.route('/predict', methods=['GET'])
# def predict():
#     """
#     Returns just one predicted price using the last 60 days.
#     """
#     df = yf.download("ETH-USD", period="70d")
#     data = df['Close'].values.reshape(-1, 1)
#     scaled_data = scaler.transform(data)
#     window_size = 60
#     X_input = scaled_data[-window_size:]
#     X_input = np.reshape(X_input, (1, window_size, 1))

#     predicted_scaled = model.predict(X_input)
#     predicted_price = scaler.inverse_transform(predicted_scaled)
#     price = round(float(predicted_price[0][0]), 2)
#     return jsonify({"predictedPrice": price})

from flask import Flask, jsonify, send_file
from flask_cors import CORS
import yfinance as yf
import numpy as np
import pickle
from tensorflow.keras.models import load_model
import io
import matplotlib
matplotlib.use('Agg')  # Force the Agg backend for non-interactive use
import matplotlib.pyplot as plt

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing

# Load the trained LSTM model and scaler (trained with 5 features: Open, High, Low, Close, Volume)
model = load_model("eth_price_model.h5")
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

@app.route('/predict', methods=['GET'])
def predict():
    """
    Fetch the latest 100 days of ETH-USD data (with 5 features),
    use the last 90 days as input to predict the next 'Close' price,
    and return the predicted price as JSON.
    """
    # Download data with auto_adjust disabled to get raw columns
    df = yf.download("ETH-USD", period="100d", auto_adjust=False)
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    df = df[required_columns].dropna()
    data = df.values  # Expected shape: (n_days, 5)
    
    # Scale the data using the scaler fitted during training
    scaled_data = scaler.transform(data)
    
    # Use the last 90 days as input; ensure we have enough data
    window_size = 90
    if len(scaled_data) < window_size:
        return jsonify({"error": f"Not enough data. Required {window_size} days, got {len(scaled_data)}."}), 500
    X_input = scaled_data[-window_size:]  # Shape: (90, 5)
    X_input = np.reshape(X_input, (1, window_size, 5))
    
    # Predict the next 'Close' price (model output is a single value)
    predicted_scaled = model.predict(X_input)
    
    # Create a dummy array with 5 features, place predicted value in the 'Close' index (index 3)
    dummy = np.zeros((1, 5))
    dummy[0, 3] = predicted_scaled[0, 0]
    
    # Inverse transform to get the actual predicted price
    predicted_full = scaler.inverse_transform(dummy)
    predicted_close = predicted_full[0, 3]
    price = round(float(predicted_close), 2)
    
    return jsonify({"predictedPrice": price})

@app.route('/graph', methods=['GET'])
def graph():
    """
    Generates a graph showing the last 100 days of the 'Close' prices along with a horizontal
    line indicating the predicted 'Close' price. Returns the graph image as a PNG.
    """
    # Download data
    df = yf.download("ETH-USD", period="100d", auto_adjust=False)
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    df = df[required_columns].dropna()
    data = df.values  # Shape: (n_days, 5)
    
    # Scale the data
    scaled_data = scaler.transform(data)
    
    # Use a window size of 90 days for prediction
    window_size = 90
    if len(scaled_data) < window_size:
        return jsonify({"error": f"Not enough data for graph. Required {window_size} days, got {len(scaled_data)}."}), 500
    X_input = scaled_data[-window_size:]
    X_input = np.reshape(X_input, (1, window_size, 5))
    predicted_scaled = model.predict(X_input)
    
    # Prepare dummy array for inverse transformation of the predicted 'Close'
    dummy = np.zeros((1, 5))
    dummy[0, 3] = predicted_scaled[0, 0]
    predicted_full = scaler.inverse_transform(dummy)
    predicted_price = round(float(predicted_full[0, 3]), 2)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    # Plot the actual 'Close' prices for the last 100 days
    ax.plot(df['Close'][-100:], label='Real ETH Close', color='blue')
    # Plot a horizontal line for the predicted price
    ax.axhline(y=predicted_price, color='red', linestyle='--', label=f'Predicted Close: ${predicted_price}')
    ax.set_title('Ethereum Price (Last 100 Days) & Prediction')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price (USD)')
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the plot to a BytesIO buffer
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    plt.close(fig)  # Close the figure to free memory
    
    return send_file(buf, mimetype='image/png')

if __name__ == "__main__":
    app.run(debug=True, port=5000)
