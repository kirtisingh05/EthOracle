# import yfinance as yf
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import MinMaxScaler
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, LSTM
# from tensorflow.keras.callbacks import EarlyStopping
# import pickle

# # Download historical Ethereum (ETH-USD) data from Yahoo Finance
# df = yf.download("ETH-USD", start="2020-01-01", end="2023-12-31")
# data = df['Close'].values.reshape(-1, 1)

# # Scale the data to the range (0, 1)
# scaler = MinMaxScaler(feature_range=(0, 1))
# scaled_data = scaler.fit_transform(data)

# # Function to create dataset using a sliding window
# def create_dataset(dataset, window_size=60):
#     X, y = [], []
#     for i in range(window_size, len(dataset)):
#         X.append(dataset[i-window_size:i, 0])
#         y.append(dataset[i, 0])
#     return np.array(X), np.array(y)

# window_size = 60
# X, y = create_dataset(scaled_data, window_size)
# # Reshape X for LSTM: (samples, timesteps, features)
# X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# # Split data into training and testing sets (80% train, 20% test)
# train_size = int(len(X) * 0.8)
# X_train, X_test = X[:train_size], X[train_size:]
# y_train, y_test = y[:train_size], y[train_size:]

# # Build the LSTM model
# model = Sequential()
# model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
# model.add(LSTM(units=50))
# model.add(Dense(1))
# model.compile(optimizer='adam', loss='mean_squared_error')

# # Train the model with early stopping to avoid overfitting
# early_stop = EarlyStopping(monitor='loss', patience=10)
# history = model.fit(X_train, y_train, epochs=50, batch_size=32,
#                     validation_data=(X_test, y_test),
#                     callbacks=[early_stop])

# # (Optional) Plot real vs predicted prices
# predictions = model.predict(X_test)
# predictions = scaler.inverse_transform(predictions)
# real_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

# plt.figure(figsize=(10,6))
# plt.plot(real_prices, color='blue', label='Real Ethereum Price')
# plt.plot(predictions, color='red', label='Predicted Ethereum Price')
# plt.title('Ethereum Price Prediction')
# plt.xlabel('Time')
# plt.ylabel('Price (USD)')
# plt.legend()
# plt.show()

# # Save the trained model and scaler for future prediction use
# model.save("eth_price_model.h5")
# with open("scaler.pkl", "wb") as f:
#     pickle.dump(scaler, f)

# print("Model and scaler saved successfully!")
"""
train_model.py

Train an improved LSTM model for Ethereum price prediction using multiple features
(Open, High, Low, Close, Volume). Adds dropout, more units, and a lower learning rate.
"""

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import pickle

# ---------------------------
# 1. Download Data (Multiple Features)
# ---------------------------
# Let's fetch 2 years of data for a broader historical context
df = yf.download("ETH-USD", start="2023-01-01", end="2025-01-01")

# Keep columns: Open, High, Low, Close, Volume
df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()

# Convert to numpy array
dataset = df.values  # shape: (num_days, 5)

# ---------------------------
# 2. Scale the Data
# ---------------------------
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

# We'll save the scaler at the end
num_features = scaled_data.shape[1]  # should be 5

# ---------------------------
# 3. Create Sliding Window Dataset
# ---------------------------
def create_dataset(data, window_size=90):
    X, y = [], []
    # We'll predict the Close price, which is index 3 in [Open,High,Low,Close,Volume]
    close_idx = 3
    for i in range(window_size, len(data)):
        # From i-window_size to i is our window
        X.append(data[i-window_size:i, :])  # shape: (window_size, 5)
        # We'll predict the 'Close' feature of the i-th day
        y.append(data[i, close_idx])
    return np.array(X), np.array(y)

window_size = 90
X, y = create_dataset(scaled_data, window_size)

# Reshape X for LSTM: (samples, timesteps, features)
# shape: (num_samples, 90, 5)
print("X shape:", X.shape)
print("y shape:", y.shape)

# ---------------------------
# 4. Split into Train/Test
# ---------------------------
train_ratio = 0.8
train_size = int(len(X) * train_ratio)

X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

print("Train size:", X_train.shape, y_train.shape)
print("Test size:", X_test.shape, y_test.shape)

# ---------------------------
# 5. Build a More Complex LSTM Model
# ---------------------------
model = Sequential()

# 1st LSTM layer
model.add(LSTM(units=128, return_sequences=True, input_shape=(window_size, num_features)))
model.add(Dropout(0.2))

# 2nd LSTM layer
model.add(LSTM(units=128))
model.add(Dropout(0.2))

# Final Dense layer for single-value output
model.add(Dense(1))

# Use a lower learning rate for stable training
optimizer = Adam(learning_rate=1e-4)

model.compile(optimizer=optimizer, loss='mean_squared_error')

# ---------------------------
# 6. Train the Model
# ---------------------------
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stop],
    verbose=1
)

# ---------------------------
# 7. Evaluate & Visualize
# ---------------------------
# Predictions on the test set
predictions = model.predict(X_test)

# We'll inverse transform just the 'Close' dimension
# shape of predictions is (num_test_samples, 1)
# We need to reconstruct a shape that matches (num_test_samples, 5) to inverse transform
# Then slice the 'Close' feature
test_pred_full = np.zeros((len(predictions), num_features))  # placeholder
test_pred_full[:, 3] = predictions[:, 0]  # put predictions in the 'Close' index
# everything else can be zeros or any valid scaled value
# We'll inverse transform
predictions_rescaled = scaler.inverse_transform(test_pred_full)[:, 3]

# Similarly, we must reconstruct the real 'Close' from y_test
# shape: (num_test_samples,)
test_real_full = np.zeros((len(y_test), num_features))
test_real_full[:, 3] = y_test
real_rescaled = scaler.inverse_transform(test_real_full)[:, 3]

# Plot real vs. predicted
plt.figure(figsize=(10,6))
plt.plot(real_rescaled, color='blue', label='Real ETH Close')
plt.plot(predictions_rescaled, color='red', label='Predicted ETH Close')
plt.title('ETH Price Prediction (Test Set)')
plt.xlabel('Days')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()

# ---------------------------
# 8. Save Model & Scaler
# ---------------------------
model.save("eth_price_model.h5")

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Training complete. Model and scaler saved.")
