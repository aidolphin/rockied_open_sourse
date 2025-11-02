import React from 'react';

const ResultCard = ({ result }) => {
  if (!result) return null;

  return (
    <div className="result-card">
      <h2>Rock Identification Result</h2>
      <div className="result-content">
        <h3>{result.name}</h3>
        <p className="confidence">Confidence: {(result.confidence * 100).toFixed(2)}%</p>
        <p className="description">{result.description}</p>
        {result.properties && (
          <div className="properties">
            <h4>Properties:</h4>
            <ul>
              {Object.entries(result.properties).map(([key, value]) => (
                <li key={key}>
                  <strong>{key}:</strong> {value}
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </div>
  );
};

export default ResultCard;
