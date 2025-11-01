# RockID - Rock Identification Web Application

A web application for identifying rocks using machine learning. Users can upload images, capture photos via webcam, or provide image URLs to identify different types of rocks.

## Project Structure

```
rockid-webapp/
├── frontend/          # React frontend
│   ├── public/
│   ├── src/
│   │   ├── components/
│   │   │   ├── WebcamCapture.jsx
│   │   │   ├── ImageUpload.jsx
│   │   │   ├── UrlInput.jsx
│   │   │   └── ResultCard.jsx
│   │   ├── App.jsx
│   │   └── model/
│   └── package.json
├── backend/           # Flask backend
│   ├── app.py
│   ├── models/
│   ├── data/
│   └── requirements.txt
└── README.md
```

## Features

- 📸 Webcam capture for real-time rock identification
- 📁 Image upload from local files
- 🔗 URL-based image analysis
- 🧠 CNN-based rock classification
- 📊 Confidence scores and detailed rock information

## Setup

### Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

## Rock Types Supported

- Granite (Igneous)
- Basalt (Igneous)
- Limestone (Sedimentary)
- Sandstone (Sedimentary)
- Marble (Metamorphic)
- Slate (Metamorphic)

## API Endpoints

- `POST /api/classify` - Classify a rock image
- `GET /api/health` - Check API health status

## Technologies Used

- **Frontend**: React, Vite
- **Backend**: Flask, PyTorch
- **ML Model**: Convolutional Neural Network (CNN)

## Notes

- The ML model file (`rock_cnn.pth`) needs to be trained separately
- TensorFlow Lite model (`rock_classifier.tflite`) can be added for browser-based inference
- Update CORS settings in production

## License

MIT
#rockid-webapp
#rockid-webapp
#rockid-webapp
#rockid-webapp
