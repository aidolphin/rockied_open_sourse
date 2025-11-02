import React, { useState } from 'react';

const UrlInput = ({ onSubmit }) => {
  const [url, setUrl] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    if (url) {
      onSubmit(url);
    }
  };

  return (
    <div className="url-input">
      <form onSubmit={handleSubmit}>
        <input
          type="url"
          placeholder="Enter image URL"
          value={url}
          onChange={(e) => setUrl(e.target.value)}
        />
        <button type="submit">Analyze</button>
      </form>
    </div>
  );
};

export default UrlInput;
