from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from pathlib import Path

app = FastAPI()

# Allow frontend access
origins = ["http://localhost", "http://localhost:3000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
MODEL_PATH = (Path(__file__).resolve().parent.parent / "models" / "potatoes.keras").resolve()
print("Loading model from:", MODEL_PATH)
MODEL = tf.keras.models.load_model(str(MODEL_PATH))

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

# ---------- Serve Frontend ----------
app.mount("/images", StaticFiles(directory="images"), name="images")

@app.get("/")
async def home():
    return FileResponse("templates/home.html")

# ---------- API ----------
@app.get("/ping")
async def ping():
    return {"message": "Hello, I am alive"}

def read_file_as_image(data) -> np.ndarray:
    img = Image.open(BytesIO(data)).convert("RGB").resize((256, 256))
    return np.array(img, dtype=np.float32)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")
    if file.content_type not in ("image/jpeg", "image/png", "image/webp"):
        raise HTTPException(status_code=400, detail=f"Unsupported content type: {file.content_type}")

    try:
        image = read_file_as_image(await file.read())
        img_batch = np.expand_dims(image, 0)  # (1, 256, 256, 3)
        preds = MODEL.predict(img_batch, verbose=0)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    predicted_class = CLASS_NAMES[int(np.argmax(preds))]
    confidence = float(np.max(preds))
    return {"class": predicted_class, "confidence": confidence}

# ---------- Run ----------
if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000, reload=True)
