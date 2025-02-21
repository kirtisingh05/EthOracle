export async function getPricePrediction() {
  const response = await fetch('http://localhost:5000/predict');
  if (!response.ok) {
    throw new Error('Failed to fetch prediction');
  }
  return await response.json();
}


export async function getGraphData() {
  const response = await fetch('http://localhost:5000/graph');
  if (!response.ok) {
    throw new Error('Failed to fetch graph data');
  }
  return response.json();
}
