import os
import io
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import tensorflow as tf

app = FastAPI(title="NeuroScan AI - Brain Tumor Detection")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Find index.html (supports both flat and subfolder layouts)
INDEX_HTML_PATH = os.path.join(BASE_DIR, "templates", "index.html")
if not os.path.exists(INDEX_HTML_PATH):
    INDEX_HTML_PATH = os.path.join(BASE_DIR, "index.html")

# Find sample_images directory
SAMPLE_DIR = os.path.join(BASE_DIR, "sample_images")
if os.path.exists(SAMPLE_DIR):
    app.mount("/sample_images", StaticFiles(directory=SAMPLE_DIR), name="sample_images")


# ==========================================
# MODEL CONFIGURATION
# ==========================================
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]

CLASS_LABELS = {
    "glioma": "Glioma Tumor",
    "meningioma": "Meningioma Tumor",
    "notumor": "Healthy Brain (No Tumor)",
    "pituitary": "Pituitary Tumor"
}

CLASS_INSIGHTS = {
    "glioma": "Gliomas originate in glial cells supporting neurons. Depending on staging, they vary from low-grade to aggressive glioblastoma. Neurological and oncological consultation is strongly recommended.",
    "meningioma": "Meningiomas arise from the meninges covering the brain and spinal cord. Most are slow-growing and benign. Regular radiological monitoring or surgical assessment is typical.",
    "pituitary": "Pituitary tumors develop in the pituitary gland affecting endocrine control. The majority are benign pituitary adenomas requiring endocrine and neurosurgical review.",
    "notumor": "The MRI scan demonstrates normal neuroanatomical structures without distinct neoplastic lesions or tumor masses in the scanned slices."
}

IMG_SIZE = (224, 224)

# Load Model
MODEL_PATH = os.path.join(BASE_DIR, "best_cnn_model.keras")
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(BASE_DIR, "models", "best_cnn_model.keras")

print(f"Loading CNN model from: {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully!")

# ==========================================
# ROUTES
# ==========================================

@app.get("/", response_class=HTMLResponse)
async def serve_home():
    if os.path.exists(INDEX_HTML_PATH):
        with open(INDEX_HTML_PATH, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse("<h1>NeuroScan AI</h1><p>Frontend template not found. Backend API ready at /predict.</p>")

@app.post("/predict")
async def predict_tumor(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image = image.resize(IMG_SIZE)
        
        img_array = np.array(image, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        preds = model.predict(img_array, verbose=0)[0]
        top_idx = int(np.argmax(preds))
        top_class = CLASS_NAMES[top_idx]
        confidence = float(preds[top_idx]) * 100.0
        
        probabilities = []
        for i, name in enumerate(CLASS_NAMES):
            probabilities.append({
                "class_name": name,
                "display_name": CLASS_LABELS[name],
                "probability": float(preds[i])
            })
            
        return JSONResponse(content={
            "prediction_class": top_class,
            "prediction_label": CLASS_LABELS[top_class],
            "confidence": round(confidence, 2),
            "insight": CLASS_INSIGHTS[top_class],
            "probabilities": probabilities
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
