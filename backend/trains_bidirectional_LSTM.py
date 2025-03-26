import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
import pickle

# 1. Download historical Ethereum (ETH-USD) data
df = yf.download("ETH-USD", start="2020-01-01", end="2023-12-31")
# Use only the 'Close' price
data = df['Close'].values.reshape(-1, 1)

# 2. Scale the data to the range (0,1)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# 3. Create a sliding window dataset
def create_dataset(dataset, window_size=60):
    X, y = [], []
    for i in range(window_size, len(dataset)):
        X.append(dataset[i-window_size:i, 0])
        y.append(dataset[i, 0])
    return np.array(X), np.array(y)

window_size = 60
X, y = create_dataset(scaled_data, window_size)
# Reshape X to be [samples, time steps, features]
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# 4. Split the data into training and testing sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 5. Build a Bidirectional LSTM model
model = Sequential()
# First Bidirectional LSTM layer with 50 units (for each direction)
model.add(Bidirectional(LSTM(units=50, return_sequences=True), input_shape=(X_train.shape[1], 1)))
# Second Bidirectional LSTM layer with 50 units (for each direction)
model.add(Bidirectional(LSTM(units=50)))
# Final Dense layer for output
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

# 6. Train the model with early stopping
early_stop = EarlyStopping(monitor='loss', patience=10)
history = model.fit(X_train, y_train, epochs=50, batch_size=32,
                    validation_data=(X_test, y_test),
                    callbacks=[early_stop])

# 7. Evaluate the model on test data
predictions = model.predict(X_test)
# Inverse transform predictions and real values
predictions = scaler.inverse_transform(predictions)
real_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

# 8. Plot the results
plt.figure(figsize=(10, 6))
plt.plot(real_prices, color='blue', label='Real Ethereum Price')
plt.plot(predictions, color='red', label='Predicted Ethereum Price')
plt.title('Ethereum Price Prediction using Bidirectional LSTM')
plt.xlabel('Time')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()

# 9. Save the trained model and scaler for later use (e.g., in your Flask backend)
model.save("bidirectional_eth_price_model.h5")
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Bidirectional LSTM model and scaler saved successfully!")
