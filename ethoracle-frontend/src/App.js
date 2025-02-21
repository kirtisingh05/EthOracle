import React, { useState, useEffect } from 'react';
import './App.css';
import { getPricePrediction } from './api';

function App() {
  const [predictedPrice, setPredictedPrice] = useState(null);
  const [loadingPrediction, setLoadingPrediction] = useState(true);
  const [errorPrediction, setErrorPrediction] = useState(null);

  const [graphUrl, setGraphUrl] = useState(null);
  const [loadingGraph, setLoadingGraph] = useState(true);
  const [errorGraph, setErrorGraph] = useState(null);

  // Fetch predicted price from the backend (/predict endpoint)
  useEffect(() => {
    async function fetchPrediction() {
      try {
        const data = await getPricePrediction();
        setPredictedPrice(data.predictedPrice);
      } catch (err) {
        setErrorPrediction(err.message);
      } finally {
        setLoadingPrediction(false);
      }
    }
    fetchPrediction();
  }, []);

  // Fetch graph image from the backend (/graph endpoint)
  useEffect(() => {
    async function fetchGraph() {
      try {
        const response = await fetch('http://localhost:5000/graph');
        if (!response.ok) {
          throw new Error('Failed to fetch graph');
        }
        const blob = await response.blob();
        const imageUrl = URL.createObjectURL(blob);
        setGraphUrl(imageUrl);
      } catch (err) {
        setErrorGraph(err.message);
      } finally {
        setLoadingGraph(false);
      }
    }
    fetchGraph();
  }, []);

  return (
    <div className="App">
      <header className="App-header">
        <h1>Ethereum Price Prediction</h1>
        <div className="prediction-section">
          {loadingPrediction ? (
            <p>Loading prediction...</p>
          ) : errorPrediction ? (
            <p>Error: {errorPrediction}</p>
          ) : (
            <h2>Predicted Ethereum Price: ${predictedPrice}</h2>
          )}
          <p>Ethereum price to be at the close of the next trading day</p>
        </div>
        <div className="graph-section" style={{ marginTop: '20px' }}>
          <h3>Prediction Graph</h3>
          {loadingGraph ? (
            <p>Loading graph...</p>
          ) : errorGraph ? (
            <p>Error: {errorGraph}</p>
          ) : graphUrl ? (
            <img src={graphUrl} alt="Ethereum Price Graph" style={{ maxWidth: '100%', borderRadius: '8px' }} />
          ) : null}
        </div>
      </header>
    </div>
  );
}

export default App;
