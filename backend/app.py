import os
from flask import Flask, request, jsonify
from flask_cors import CORS
try:
    import torch
    import torch.nn as nn
except Exception as e:
    # Allow the API to run without PyTorch installed.
    # The app will fall back to a simple heuristic classifier when the trained model
    # or torch isn't available.
    torch = None
    nn = None
    print(f"PyTorch not available: {e}. The server will run using the fallback classifier.")
try:
    from PIL import Image
except Exception:
    Image = None
import io
import json
import requests

app = Flask(__name__)
CORS(app)

# Load rock data
with open('data/rocks.json', 'r') as f:
    rocks_data = json.load(f)

# Load model
if nn is not None:
    class RockCNN(nn.Module):
        def __init__(self, num_classes):
            super(RockCNN, self).__init__()
            self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
            self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(128 * 28 * 28, 512)
            self.fc2 = nn.Linear(512, num_classes)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.5)

        def forward(self, x):
            x = self.pool(self.relu(self.conv1(x)))
            x = self.pool(self.relu(self.conv2(x)))
            x = self.pool(self.relu(self.conv3(x)))
            x = x.view(-1, 128 * 28 * 28)
            x = self.dropout(self.relu(self.fc1(x)))
            x = self.fc2(x)
            return x
else:
    # Dummy placeholder so module-level references to RockCNN don't break when torch
    # isn't available. Attempting to instantiate this will raise a helpful error.
    class RockCNN:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch is not installed; RockCNN is unavailable.")

model = None
model_loaded = False
model_path = 'models/rock_cnn.pth'
# Only attempt to load the PyTorch model if torch is available and the file exists
if torch is not None and os.path.exists(model_path):
    try:
        model = RockCNN(num_classes=len(rocks_data))
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        model_loaded = True
        print(f"Loaded model from {model_path}")
    except Exception as e:
        print(f"Failed to load model: {e}")
        model = None
        model_loaded = False
else:
    if torch is None:
        print("PyTorch not installed — running without trained model. Fallback classifier enabled.")
    else:
        print(f"Model file not found at {model_path}. Continuing without model. Run training to create it.")

def preprocess_image(image):
    image = image.resize((224, 224))
    image = torch.tensor(list(image.getdata())).float()
    image = image.view(224, 224, 3).permute(2, 0, 1)
    image = image / 255.0
    return image.unsqueeze(0)


def simple_fallback_classify(image):
    """A very small heuristic classifier used when the trained model is not available.
    It analyses average brightness and basic color channels to pick a likely rock.
    This is not a replacement for a trained model but keeps the API usable for demos.
    """
    img = image.resize((100, 100))
    pixels = list(img.getdata())
    r = sum([p[0] for p in pixels]) / len(pixels)
    g = sum([p[1] for p in pixels]) / len(pixels)
    b = sum([p[2] for p in pixels]) / len(pixels)
    brightness = (r + g + b) / 3

    # Very simple heuristic rules
    if brightness < 80 and (r < 100 and g < 100 and b < 120):
        return 1, 0.6  # Basalt
    if brightness > 180 and (r > 180 and g > 180 and b > 180):
        return 4, 0.6  # Marble / very light
    if r > 140 and g < 130:
        return 3, 0.55  # Sandstone (reddish)
    if abs(r - g) < 25 and abs(g - b) < 25 and brightness > 110:
        return 0, 0.6  # Granite-like (mixed colors)

    # default fallback
    return 0, 0.5

@app.route('/api/classify', methods=['POST'])
def classify():
    try:
        # Handle image upload or URL
        if 'image' in request.files:
            image_file = request.files['image']
            image = Image.open(io.BytesIO(image_file.read())).convert('RGB')
        elif 'url' in request.form:
            url = request.form['url']
            response = requests.get(url)
            image = Image.open(io.BytesIO(response.content)).convert('RGB')
        else:
            return jsonify({'error': 'No image provided'}), 400

        # Preprocess and classify
        if model_loaded and model is not None:
            image_tensor = preprocess_image(image)
            with torch.no_grad():
                output = model(image_tensor)
                probabilities = torch.nn.functional.softmax(output, dim=1)
                confidence, predicted = torch.max(probabilities, 1)

            rock_id = predicted.item()
            rock_info = rocks_data[rock_id]

            return jsonify({
                'name': rock_info['name'],
                'confidence': float(confidence.item()),
                'description': rock_info['description'],
                'properties': rock_info['properties']
            })
        else:
            # Use lightweight fallback heuristic classifier
            rock_id, conf = simple_fallback_classify(image)
            rock_info = rocks_data[rock_id]
            return jsonify({
                'name': rock_info['name'],
                'confidence': float(conf),
                'description': rock_info['description'],
                'properties': rock_info['properties'],
                'note': 'Used simple fallback classifier because trained model is not available.'
            })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    # Allow overriding the port via the PORT environment variable (default 5001)
    port = int(os.environ.get('PORT', 5001))
    app.run(debug=True, port=port)
