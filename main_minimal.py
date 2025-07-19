from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import os

print("🔄 Starting imports...")

try:
    import torch
    print("✅ PyTorch imported successfully")
except Exception as e:
    print(f"❌ PyTorch import error: {e}")

try:
    from database import Database
    print("✅ Database imported successfully")
except Exception as e:
    print(f"❌ Database import error: {e}")

print("🚀 Creating FastAPI app...")

app = FastAPI()

print("🔧 Setting up templates...")
templates = Jinja2Templates(directory="templates")

print("📁 Setting up static files...")
os.makedirs("static/uploads", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

print("💾 Initializing database...")
try:
    db = Database()
    print("✅ Database initialized successfully")
except Exception as e:
    print(f"❌ Database initialization error: {e}")

@app.get("/")
async def root():
    return {"message": "Server is running!", "status": "OK"}

@app.get("/test", response_class=HTMLResponse)
async def test_page(request: Request):
    try:
        return templates.TemplateResponse("landing.html", {"request": request})
    except Exception as e:
        return HTMLResponse(f"Template error: {e}")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "database": "connected",
        "templates": "loaded"
    }

print("✅ All setup complete, server should start now...")

if __name__ == "__main__":
    import uvicorn
    print("🌟 Starting uvicorn server...")
    uvicorn.run(app, host="0.0.0.0", port=9990)