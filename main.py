from fastapi import FastAPI, File, UploadFile, Request, Form, HTTPException, Depends, Cookie
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from torchvision import models, transforms
import torch
from PIL import Image
import io
import uvicorn
import os
import uuid
from datetime import datetime, timedelta
import jwt
from typing import Optional
from model import create_model
from database import Database

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

# JWT Secret key (in production, use environment variable)
SECRET_KEY = "your-secret-key-here-change-in-production"
ALGORITHM = "HS256"

app = FastAPI()
templates = Jinja2Templates(directory="templates")
security = HTTPBearer(auto_error=False)

# Add startup event
@app.on_event("startup")
async def startup_event():
    print("🚀 Application starting up...")
    print(f"💾 Database initialized")
    print(f"🔧 Device: {device}")
    print("✅ Server ready!")

# Add health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "device": device,
        "model_loaded": model is not None
    }

# Create uploads directory if it doesn't exist
os.makedirs("static/uploads", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize database
db = Database()

# Model will be loaded lazily
model = None
transform = None

def load_model():
    """Load the model lazily when first needed"""
    global model, transform
    if model is None:
        try:
            print("Loading PyTorch model...")
            output_shape = 7
            model = create_model(output_shape=output_shape, device=device)
            
            # Check if model file exists
            model_path = "efficientnetb3_model.pth"
            if not os.path.exists(model_path):
                print(f"ERROR: Model file '{model_path}' not found!")
                print("Please ensure the .pth model file is in the project directory")
                return False
            
            model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
            model.eval()
            
            # Initialize transform
            image_size = (224, 224)
            transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
            
            print("Model loaded successfully!")
            return True
            
        except Exception as e:
            print(f"ERROR loading model: {e}")
            return False
    return True

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(hours=24)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def get_token_from_cookie(request: Request) -> Optional[str]:
    """Extract token from cookie"""
    token = request.cookies.get("access_token")
    if token and token.startswith("Bearer "):
        return token[7:]  # Remove "Bearer " prefix
    return None

def verify_token(request: Request):
    token = get_token_from_cookie(request)
    if not token:
        return None
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: int = payload.get("user_id")
        if user_id is None:
            return None
        return user_id
    except jwt.PyJWTError:
        return None

def get_current_user(request: Request):
    user_id = verify_token(request)
    if user_id is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    user = db.get_user_by_id(user_id)
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")
    return user

def get_current_user_optional(request: Request):
    user_id = verify_token(request)
    if user_id is None:
        return None
    return db.get_user_by_id(user_id)

def get_admin_user(request: Request):
    user = get_current_user(request)
    if not user.get('is_admin', False):
        raise HTTPException(status_code=403, detail="Admin access required")
    return user

@app.get("/", response_class=HTMLResponse)
async def landing_page(request: Request):
    current_user = get_current_user_optional(request)
    if current_user:
        return RedirectResponse(url="/dashboard", status_code=302)
    return templates.TemplateResponse("landing.html", {"request": request})

@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    current_user = get_current_user_optional(request)
    if current_user:
        return RedirectResponse(url="/dashboard", status_code=302)
    return templates.TemplateResponse("register.html", {"request": request, "error": ""})

@app.post("/register", response_class=HTMLResponse)
async def register_user(request: Request, email: str = Form(...), password: str = Form(...), confirm_password: str = Form(...)):
    if password != confirm_password:
        return templates.TemplateResponse("register.html", {"request": request, "error": "Passwords do not match"})
    
    if len(password) < 6:
        return templates.TemplateResponse("register.html", {"request": request, "error": "Password must be at least 6 characters long"})
    
    success = db.create_user(email, password)
    if not success:
        return templates.TemplateResponse("register.html", {"request": request, "error": "Email already exists"})
    
    return templates.TemplateResponse("register.html", {"request": request, "success": "Account created successfully! Please login."})

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    current_user = get_current_user_optional(request)
    if current_user:
        return RedirectResponse(url="/dashboard", status_code=302)
    return templates.TemplateResponse("login.html", {"request": request, "error": ""})

@app.post("/login", response_class=HTMLResponse)
async def login_user(request: Request, email: str = Form(...), password: str = Form(...)):
    user_id = db.verify_user(email, password)
    if not user_id:
        return templates.TemplateResponse("login.html", {"request": request, "error": "Invalid email or password"})
    
    access_token = create_access_token(data={"user_id": user_id})
    response = RedirectResponse(url="/dashboard", status_code=302)
    response.set_cookie(
        key="access_token", 
        value=f"Bearer {access_token}", 
        httponly=True, 
        max_age=86400,
        secure=False,  # Set to True in production with HTTPS
        samesite="lax"
    )
    return response

@app.get("/logout")
async def logout():
    response = RedirectResponse(url="/", status_code=302)
    response.delete_cookie("access_token")
    return response

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    current_user = get_current_user(request)
    return templates.TemplateResponse("dashboard.html", {"request": request, "user": current_user, "prediction": ""})

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    current_user = get_current_user(request)
    
    # Load model if not already loaded
    if not load_model():
        return templates.TemplateResponse("dashboard.html", {
            "request": request,
            "user": current_user,
            "prediction": "Error: Model not available. Please contact administrator.",
            "confidence": "",
            "image_path": ""
        })
    
    try:
        contents = await file.read()

        # Generate unique filename
        file_extension = os.path.splitext(file.filename)[1]
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        image_path = f"static/uploads/{unique_filename}"
        
        with open(image_path, "wb") as f:
            f.write(contents)

        image = Image.open(io.BytesIO(contents)).convert("RGB")
        img_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.nn.functional.softmax(outputs[0], dim=0)
            pred_idx = torch.argmax(probs).item()
            confidence = probs[pred_idx].item()
            prediction = CLASS_NAMES[pred_idx]

        # Save prediction to database
        db.save_prediction(current_user['id'], unique_filename, prediction, confidence)

        return templates.TemplateResponse("dashboard.html", {
            "request": request,
            "user": current_user,
            "prediction": f"Predicted Disease: {prediction}",
            "confidence": f"Confidence: {confidence:.2%}",
            "image_path": "/" + image_path
        })
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return templates.TemplateResponse("dashboard.html", {
            "request": request,
            "user": current_user,
            "prediction": f"Error during prediction: {str(e)}",
            "confidence": "",
            "image_path": ""
        })

@app.get("/history", response_class=HTMLResponse)
async def history(request: Request):
    current_user = get_current_user(request)
    predictions = db.get_user_predictions(current_user['id'])
    return templates.TemplateResponse("history.html", {
        "request": request, 
        "user": current_user, 
        "predictions": predictions
    })

@app.post("/delete-prediction/{prediction_id}")
async def delete_prediction(request: Request, prediction_id: int):
    current_user = get_current_user(request)
    success = db.delete_prediction(prediction_id, current_user['id'])
    if success:
        return {"success": True}
    else:
        raise HTTPException(status_code=404, detail="Prediction not found")

@app.get("/account", response_class=HTMLResponse)
async def account_page(request: Request):
    current_user = get_current_user(request)
    return templates.TemplateResponse("account.html", {"request": request, "user": current_user})

@app.post("/delete-account", response_class=HTMLResponse)
async def delete_account(request: Request, password: str = Form(...)):
    current_user = get_current_user(request)
    success = db.delete_user_account(current_user['id'], password)
    if success:
        response = RedirectResponse(url="/?deleted=1", status_code=302)
        response.delete_cookie("access_token")
        return response
    else:
        return templates.TemplateResponse("account.html", {
            "request": request, 
            "user": current_user, 
            "error": "Invalid password"
        })

# Admin routes
@app.get("/admin", response_class=HTMLResponse)
async def admin_dashboard(request: Request):
    admin_user = get_admin_user(request)
    users = db.get_all_users()
    predictions = db.get_all_predictions()
    
    stats = {
        'total_users': len(users),
        'total_predictions': len(predictions),
        'recent_users': len([u for u in users if (datetime.now() - datetime.fromisoformat(u['created_at'].replace('Z', '+00:00').replace('+00:00', ''))).days <= 7]),
        'recent_predictions': len([p for p in predictions if (datetime.now() - datetime.fromisoformat(p['created_at'].replace('Z', '+00:00').replace('+00:00', ''))).days <= 7])
    }
    
    return templates.TemplateResponse("admin.html", {
        "request": request,
        "admin": admin_user,
        "users": users,
        "predictions": predictions,
        "stats": stats
    })

@app.get("/admin-login", response_class=HTMLResponse)
async def admin_login_page(request: Request):
    current_user = get_current_user_optional(request)
    if current_user and current_user.get('is_admin'):
        return RedirectResponse(url="/admin", status_code=302)
    return templates.TemplateResponse("admin_login.html", {"request": request, "error": ""})

@app.post("/admin-login", response_class=HTMLResponse)
async def admin_login(request: Request, email: str = Form(...), password: str = Form(...)):
    user_id = db.verify_user(email, password)
    if user_id:
        user = db.get_user_by_id(user_id)
        if user and user.get('is_admin'):
            access_token = create_access_token(data={"user_id": user_id})
            response = RedirectResponse(url="/admin", status_code=302)
            response.set_cookie(
                key="access_token", 
                value=f"Bearer {access_token}", 
                httponly=True, 
                max_age=86400,
                secure=False,
                samesite="lax"
            )
            return response
    
    return templates.TemplateResponse("admin_login.html", {
        "request": request, 
        "error": "Invalid admin credentials"
    })

@app.post("/admin/delete-user/{user_id}")
async def admin_delete_user(request: Request, user_id: int):
    get_admin_user(request)  # Verify admin access
    success = db.admin_delete_user(user_id)
    if success:
        return {"success": True}
    else:
        raise HTTPException(status_code=400, detail="Cannot delete user")

@app.post("/admin/delete-prediction/{prediction_id}")
async def admin_delete_prediction(request: Request, prediction_id: int):
    get_admin_user(request)  # Verify admin access
    success = db.admin_delete_prediction(prediction_id)
    if success:
        return {"success": True}
    else:
        raise HTTPException(status_code=404, detail="Prediction not found")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)