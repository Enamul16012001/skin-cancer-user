from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from torchvision import models, transforms
import torch
from PIL import Image
import io
import uvicorn
import os

# === Configuration ===
CLASS_NAMES = [
    "Actinic Keratoses and Intraepithelial Carcinoma (akiec)",
    "Basal Cell Carcinoma (bcc)",
    "Benign Keratosis-like Lesions (bkl)",
    "Dermatofibroma (df)",
    "Melanoma (mel)",
    "Melanocytic Nevi (nv)",
    "Vascular Lesions (vasc)"
]
NUM_CLASSES = len(CLASS_NAMES)
device = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# === Load Model ===
model = models.efficientnet_b3(weights=None)

output_shape = 7

#num_ftrs = model.fc.in_features
num_ftrs = model.classifier[1].in_features

#model.fc = torch.nn.Sequential(
model.classifier = torch.nn.Sequential(
    #torch.nn.Dropout(p=0.2, inplace=True),
    torch.nn.Linear(in_features=num_ftrs,
                    out_features=512,
                    bias=True),
    torch.nn.ReLU(),
    torch.nn.Linear(in_features=512,
                    out_features=128,
                    bias=True),
    torch.nn.ReLU(),
    torch.nn.Linear(in_features=128,
                    out_features=output_shape,
                    bias=True)).to(device)


model.load_state_dict(torch.load("efficientnetb3_model.pth", map_location=torch.device('cpu')))
model.eval()

image_size = (224, 224)
# === Image Transform ===
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
# === Routes ===
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "prediction": ""})

# Mount static folder if not already
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    contents = await file.read()

    # Save to static folder for preview
    image_path = f"static/uploaded_image.jpg"
    with open(image_path, "wb") as f:
        f.write(contents)

    # Load image and predict
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)
        pred_idx = torch.argmax(probs).item()
        prediction = CLASS_NAMES[pred_idx]

    return templates.TemplateResponse("index.html", {
        "request": request,
        "prediction": f"Predicted Disease: {prediction}",
        "image_path": "/" + image_path  # pass to template
    })
