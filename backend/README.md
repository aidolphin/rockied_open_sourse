# Backend README — training and running

This backend exposes a simple Flask API at `/api/classify` and can optionally load a PyTorch model saved at `models/rock_cnn.pth`.

Fallback behavior
- If `models/rock_cnn.pth` exists the server will load it and use it for inference.
- If the model file is missing the server will use a very small heuristic fallback classifier (basic brightness/color rules) so the API remains usable for demos.

Training from Kaggle dataset
1. Download the Kaggle dataset to `backend/data/` (example using Kaggle CLI). You can use the helper script `backend/scripts/download_and_train.sh` which runs the download, prepares the dataset and starts training.

Using the helper script (recommended):

```bash
# ensure you have kaggle CLI configured with ~/.kaggle/kaggle.json
cd backend
chmod +x scripts/download_and_train.sh
./scripts/download_and_train.sh salmaneunus/rock-classification 8
```

Or run the commands manually:

```bash
# install kaggle CLI and configure API token first
kaggle datasets download -d salmaneunus/rock-classification -p backend/data --unzip
```

2. Prepare dataset in ImageFolder format with `train/` and `val/` subfolders. The dataset must have class subfolders inside `train/` and `val/`.

3. Run training:

```bash
python backend/train.py --dataset backend/data/rock-classification --out backend/models/rock_cnn.pth --epochs 5 --batch 16
```

Notes
- Training on CPU is slow. Use a GPU for reasonable training times.
- The provided `train.py` uses MobileNetV2 features and saves the best checkpoint based on validation accuracy.

Running the server

```bash
pip install -r backend/requirements.txt
python backend/app.py
```
