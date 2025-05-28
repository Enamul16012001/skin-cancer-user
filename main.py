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
from model import create_model

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

output_shape = 7

model = create_model(output_shape=output_shape, device=device)

model.load_state_dict(torch.load("efficientnetb3_model.pth", map_location=torch.device(device)))
model.eval()

image_size = (224, 224)

transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "prediction": ""})

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    contents = await file.read()

    image_path = f"static/uploaded_image.jpg"
    with open(image_path, "wb") as f:
        f.write(contents)

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
        "image_path": "/" + image_path
    })
