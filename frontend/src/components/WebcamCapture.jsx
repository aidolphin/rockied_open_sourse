import React, { useRef, useState } from 'react';

const WebcamCapture = ({ onCapture }) => {
  const videoRef = useRef(null);
  const [isStreaming, setIsStreaming] = useState(false);

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      videoRef.current.srcObject = stream;
      setIsStreaming(true);
    } catch (err) {
      console.error('Error accessing camera:', err);
    }
  };

  const captureImage = () => {
    const canvas = document.createElement('canvas');
    canvas.width = videoRef.current.videoWidth;
    canvas.height = videoRef.current.videoHeight;
    canvas.getContext('2d').drawImage(videoRef.current, 0, 0);
    canvas.toBlob((blob) => {
      onCapture(blob);
    });
  };

  return (
    <div className="webcam-capture">
      <video ref={videoRef} autoPlay playsInline />
      {!isStreaming ? (
        <button onClick={startCamera}>Start Camera</button>
      ) : (
        <button onClick={captureImage}>Capture Image</button>
      )}
    </div>
  );
};

export default WebcamCapture;
