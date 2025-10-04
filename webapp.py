import os
from flask import Flask, request, render_template, send_from_directory, jsonify
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from werkzeug.utils import secure_filename

# -------------------------------
# Siamese Network Definition
# -------------------------------
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(51200, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 128)
        )

    def forward_once(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

    def forward(self, x1, x2):
        return self.forward_once(x1), self.forward_once(x2)

# -------------------------------
# Model Loading
# -------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SiameseNetwork().to(device)
model_path = "checkpoints/siamese_best.pth"
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# -------------------------------
# Image Preprocessing
# -------------------------------
def preprocess_image(img_path):
    transform = transforms.Compose([
        transforms.Resize((155, 220)),
        transforms.ToTensor()
    ])
    img = Image.open(img_path).convert("L")
    img = transform(img).unsqueeze(0)
    return img

# -------------------------------
# Flask Setup
# -------------------------------
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    distance = None
    file1_url = None
    file2_url = None

    if request.method == "POST":
        file1 = request.files.get("file1")
        file2 = request.files.get("file2")

        if not file1 or not file2:
            return jsonify({"error": "No files uploaded!"})

        # Save uploaded files
        file1_name = secure_filename(file1.filename)
        file2_name = secure_filename(file2.filename)

        path1 = os.path.join(app.config["UPLOAD_FOLDER"], file1_name)
        path2 = os.path.join(app.config["UPLOAD_FOLDER"], file2_name)

        file1.save(path1)
        file2.save(path2)

        # Preprocess
        img1 = preprocess_image(path1).to(device)
        img2 = preprocess_image(path2).to(device)

        # Model prediction
        with torch.no_grad():
            output1, output2 = model(img1, img2)
            distance = F.pairwise_distance(output1, output2).item()

        threshold = 1.0
        result = "MATCH (likely genuine)" if distance < threshold else "DO NOT MATCH (likely forgery)"

        # URLs for frontend preview
        file1_url = f"/uploads/{file1_name}"
        file2_url = f"/uploads/{file2_name}"

        # Always return JSON for AJAX
        return jsonify({
            "result": result,
            "distance": distance,
            "file1_url": file1_url,
            "file2_url": file2_url
        })

    # Regular GET request
    return render_template(
        "index.html",
        result=result,
        distance=distance,
        file1_url=file1_url,
        file2_url=file2_url
    )

if __name__ == "__main__":
    app.run(debug=True)
