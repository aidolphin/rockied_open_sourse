import React from 'react';

const ImageUpload = ({ onUpload }) => {
  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      onUpload(file);
    }
  };

  return (
    <div className="image-upload">
      <input
        type="file"
        accept="image/*"
        onChange={handleFileChange}
        id="file-input"
      />
      <label htmlFor="file-input">Choose Image</label>
    </div>
  );
};

export default ImageUpload;
