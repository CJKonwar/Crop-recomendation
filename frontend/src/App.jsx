// frontend/src/App.js

import React, { useState } from 'react';
import './App.css';
// The import for './index.css' has been removed to resolve a compilation error.
// In this environment, Tailwind CSS styles are often applied globally without needing this import.

function App() {
  // State to hold form data
  const [formData, setFormData] = useState({
    N: '90',
    P: '42',
    K: '43',
    temperature: '20.8',
    humidity: '82',
    ph: '6.5',
    rainfall: '202.9'
  });

  // State for predictions, errors, and loading status
  const [predictions, setPredictions] = useState([]);
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  // Handle input changes
  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prevState => ({
      ...prevState,
      [name]: value
    }));
  };

  // Handle form submission
  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    setPredictions([]);
    setError('');

    try {
      // Send a POST request to the Flask backend
      const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        // Convert form data values to numbers before sending
        body: JSON.stringify(Object.fromEntries(
          Object.entries(formData).map(([key, value]) => [key, Number(value)])
        )),
      });

      if (!response.ok) {
        throw new Error('Network response was not ok');
      }

      const result = await response.json();
      if (result.predictions) {
        setPredictions(result.predictions);
      } else if (result.error) {
        setError(`Error: ${result.error}`);
      }
    } catch (err) {
      setError('An error occurred. Is the Flask server running?');
      console.error("Fetch error:", err);
    } finally {
      setIsLoading(false);
    }
  };

  // An array to map over for creating input fields
  const inputFields = [
    { name: 'N', label: 'Nitrogen', placeholder: 'e.g., 90' },
    { name: 'P', label: 'Phosphorus', placeholder: 'e.g., 42' },
    { name: 'K', label: 'Potassium', placeholder: 'e.g., 43' },
    { name: 'temperature', label: 'Temperature (°C)', placeholder: 'e.g., 20.8' },
    { name: 'humidity', label: 'Humidity (%)', placeholder: 'e.g., 82' },
    { name: 'ph', label: 'pH of Soil', placeholder: 'e.g., 6.5' },
    { name: 'rainfall', label: 'Rainfall (mm)', placeholder: 'e.g., 202.9' },
  ];

  return (
    <div className="min-h-screen bg-gray-900 text-white flex flex-col items-center justify-center p-4 font-sans">
      <div className="w-full max-w-4xl p-8 space-y-8 bg-gray-800 rounded-2xl shadow-lg">
        <h1 className="text-3xl font-bold text-center text-green-400">
          Crop Recommendation System
        </h1>
        <p className="text-center text-gray-400">
          Enter soil and weather conditions to get the top 5 crop recommendations.
        </p>

        <form onSubmit={handleSubmit} className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {inputFields.map((field) => (
              <div key={field.name}>
                <label htmlFor={field.name} className="text-sm font-medium text-gray-300 block mb-2">
                  {field.label}
                </label>
                <input
                  type="number"
                  step="any" // Allow decimal values
                  name={field.name}
                  id={field.name}
                  value={formData[field.name]}
                  onChange={handleChange}
                  placeholder={field.placeholder}
                  required
                  className="w-full px-4 py-2 bg-gray-700 border border-gray-600 rounded-lg focus:ring-green-500 focus:border-green-500 transition"
                />
              </div>
            ))}
          </div>

          <button
            type="submit"
            disabled={isLoading}
            className="w-full py-3 px-4 bg-green-600 hover:bg-green-700 rounded-lg text-white font-semibold transition duration-300 disabled:bg-gray-500"
          >
            {isLoading ? 'Predicting...' : 'Get Recommendation'}
          </button>
        </form>

        {error && (
          <div className="mt-6 p-4 text-center bg-red-900/50 border border-red-500 rounded-lg">
            <p className="text-xl font-semibold text-red-300">{error}</p>
          </div>
        )}

        {predictions.length > 0 && (
          <div className="mt-6 p-6 bg-gray-700 rounded-lg">
            <h2 className="text-2xl font-semibold text-center mb-4 text-green-300">Top 5 Recommendations</h2>
            <div className="space-y-3">
              {predictions.map((p, index) => (
                <div key={index} className="flex items-center">
                  <span className="font-bold text-lg w-32">{p.crop}</span>
                  <div className="w-full bg-gray-600 rounded-full h-6">
                    <div
                      className="bg-green-500 h-6 rounded-full text-xs font-medium text-blue-100 text-center p-1 leading-none"
                      style={{ width: `${p.score * 100}%` }}
                    >
                      {`${(p.score * 100).toFixed(1)}%`}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;