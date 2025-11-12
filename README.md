# EthOracle: Ethereum Price Prediction Using LSTM

EthOracle is an AI-driven web application that forecasts Ethereum (ETH) prices using a Long Short-Term Memory (LSTM) neural network. The project features a Flask backend that fetches historical price data from Yahoo Finance, preprocesses it, and uses a pre-trained LSTM model to predict future prices. A React frontend consumes the backend API to display real-time predictions in a modern, responsive UI.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Folder Structure](#folder-structure)
- [Installation](#installation)
  - [Backend Setup](#backend-setup)
  - [Frontend Setup](#frontend-setup)
- [Usage](#usage)
- [Screenshots](#screenshots)

## Overview

The goal of EthOracle is to provide an accurate prediction of Ethereum prices using deep learning. The application downloads historical data from Yahoo Finance, processes it, and uses an LSTM-based neural network to predict future price movements. The predicted price is then served via a RESTful API and displayed on a sleek React frontend.

## Features

- **Data Collection:** Retrieves historical Ethereum (ETH-USD) price data from Yahoo Finance.
- **Data Preprocessing:** Uses MinMaxScaler to normalize the data.
- **Model Training:** Implements an LSTM network with early stopping to prevent overfitting.
- **Prediction:** Forecasts the next price using the trained model.
- **Visualization:** (Optional) Plots actual vs. predicted prices.
- **API Integration:** Flask backend with CORS enabled for seamless integration with the React frontend.
- **Responsive UI:** Modern React app with an attractive and animated CSS design.

## Technologies Used

- **Backend:** Python, Flask, Flask-CORS, TensorFlow/Keras, yfinance, NumPy, Pandas, Scikit-Learn, Pickle
- **Frontend:** React, HTML, CSS, JavaScript
- **Deployment:** Localhost for development (can be deployed on platforms like Heroku, Netlify, or Render)
- 
## Folder Structure
```md
EthOracle/
├── backend/
│   ├── app.py
│   ├── requirements.txt
│   ├── eth_price_model.h5
│   └── scaler.pkl
└── frontend/
    ├── public/
    │   └── index.html
    ├── src/
    │   ├── App.js
    │   ├── App.css
    │   ├── api.js
    │   └── index.js
    └── package.json
```

## Installation

### Backend Setup

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/EthOracle.git
   cd EthOracle/backend
   ```

2. **(Optional) Create a Virtual Environment:**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   # or
   source venv/bin/activate  # macOS/Linux
   ```

3. **Upgrade pip, setuptools, and wheel:**
   ```bash
   python -m pip install --upgrade pip setuptools wheel
   ```

4. **Install Dependencies:**
   ```bash
   pip install --only-binary=:all: -r requirements.txt
   ```

5. **Ensure** that `eth_price_model.h5` and `scaler.pkl` are in the backend folder.

6. **Run the Flask Server:**
   ```bash
   python app.py
   ```
   The server will start at [http://localhost:5000](http://localhost:5000).

### Frontend Setup

1. **Navigate to the Frontend Folder:**
   ```bash
   cd ../frontend
   ```

2. **Install Dependencies:**
   ```bash
   npm install
   ```

3. **Start the React App:**
   ```bash
   npm start
   ```
   The app will open in your default browser at [http://localhost:3000](http://localhost:3000).

## Usage

- **Backend API:**  
  Visit [http://localhost:5000/predict](http://localhost:5000/predict) to see a JSON response with the predicted Ethereum price.
  
- **Frontend:**  
  The React application automatically fetches the predicted price from the Flask backend and displays it on the homepage.

## Screenshots

![EthOracle Home](https://github.com/user-attachments/assets/7103e0f1-f167-4078-9951-8a3fdd368888)
![image](https://github.com/user-attachments/assets/c8ea5478-2d23-4137-ab8d-d04287f883c9)

*Example of the EthOracle React frontend displaying the predicted price.*

## Contact

For questions, suggestions, or contributions, please contact [kirtisingh0543@gmail.com](mailto:kirtisingh0543@gmail.com) or open an issue in the repository.

*Happy predicting!*
(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)