import os
from flask import Flask, request, render_template
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from werkzeug.utils import secure_filename

# -------------------------------
# Siamese Network
# -------------------------------
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            nn.Conv2d(32, 64, 5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            nn.Conv2d(64, 128, 3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2)
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
# Load model once
# -------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SiameseNetwork().to(device)
model_path = "siamese_epoch20.pth"
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# -------------------------------
# Image preprocessing
# -------------------------------
def preprocess_image(img_path):
    transform = transforms.Compose([
        transforms.Resize((155,220)),
        transforms.ToTensor()
    ])
    img = Image.open(img_path).convert("L")
    img = transform(img).unsqueeze(0)
    return img

# -------------------------------
# Flask setup
# -------------------------------
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    distance = None
    if request.method == "POST":
        if "file1" not in request.files or "file2" not in request.files:
            result = "No files uploaded!"
        else:
            file1 = request.files["file1"]
            file2 = request.files["file2"]
            path1 = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file1.filename))
            path2 = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file2.filename))
            file1.save(path1)
            file2.save(path2)

            # preprocess images
            img1 = preprocess_image(path1).to(device)
            img2 = preprocess_image(path2).to(device)

            # predict
            with torch.no_grad():
                output1, output2 = model(img1, img2)
                distance = F.pairwise_distance(output1, output2).item()

            threshold = 1.0  # adjust based on validation
            result = "✅ MATCH (likely genuine)" if distance < threshold else "❌ DO NOT MATCH (likely forgery)"

    return render_template("index.html", result=result, distance=distance)

if __name__ == "__main__":
    app.run(debug=True)
