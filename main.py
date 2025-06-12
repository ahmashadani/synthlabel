# SynthLabel Studio - Phase 1: MVP Architecture & Core Setup (Fixed)

# ------------------------------
# Backend: FastAPI with YOLOv8 & SAM Integration, Export + Docker
# ------------------------------
from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
import uvicorn
import os
import shutil
from ultralytics import YOLO
import torch
import cv2
import numpy as np
import json

try:
    from segment_anything import SamPredictor, sam_model_registry
except ImportError:
    SamPredictor = None
    sam_model_registry = None

app = FastAPI()

# CORS setup for frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
LABEL_DIR = "labels"
VIS_DIR = "visuals"
EXPORT_DIR = "exports"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(LABEL_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)
os.makedirs(EXPORT_DIR, exist_ok=True)

# Load YOLOv8 model
yolo_model = YOLO("yolov8n.pt")

# Load SAM model if available
if SamPredictor and sam_model_registry:
    sam = sam_model_registry["vit_b"](checkpoint="sam_vit_b.pth")
    sam.to(device="cuda" if torch.cuda.is_available() else "cpu")
    sam_predictor = SamPredictor(sam)
else:
    sam_predictor = None

@app.get("/")
def root():
    return {"message": "SynthLabel Studio API is up!"}

@app.post("/upload/")
async def upload_image(file: UploadFile):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    img = cv2.imread(file_path)
    results = yolo_model(img)
    detections = results[0].boxes.xyxy.cpu().numpy()
    label_data = []

    overlay = img.copy()

    for i, box in enumerate(detections):
        color = np.random.randint(0, 255, (1, 3)).tolist()[0]
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        mask = None

        if sam_predictor:
            sam_predictor.set_image(img)
            transformed_box = sam_predictor.transform.apply_boxes_torch(
                torch.tensor([box], dtype=torch.float), img.shape[:2]
            ).to(sam_predictor.device)
            masks, _, _ = sam_predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_box,
                multimask_output=False
            )
            mask = masks[0].cpu().numpy().astype(np.uint8).tolist()
            overlay = np.where(np.expand_dims(masks[0].cpu().numpy(), -1), overlay * 0.5 + np.array(color) * 0.5, overlay).astype(np.uint8)

        label_data.append({
            "bbox": box.tolist(),
            "label": "object",
            "mask": mask if mask else []
        })

    vis_path = os.path.join(VIS_DIR, f"{file.filename}_vis.jpg")
    cv2.imwrite(vis_path, overlay)

    label_path = os.path.join(LABEL_DIR, f"{file.filename}.json")
    with open(label_path, "w") as f:
        json.dump(label_data, f)

    return JSONResponse({
        "filename": file.filename,
        "num_detections": len(label_data),
        "label_path": label_path,
        "visual_path": f"/visual/{file.filename}_vis.jpg"
    })

@app.get("/visual/{filename}")
def get_visual(filename: str):
    vis_path = os.path.join(VIS_DIR, filename)
    if os.path.exists(vis_path):
        return FileResponse(vis_path, media_type="image/jpeg")
    return JSONResponse(status_code=404, content={"error": "Visualization not found"})

@app.get("/export/{filename}")
def export_yolo_format(filename: str):
    label_path = os.path.join(LABEL_DIR, f"{filename}.json")
    export_path = os.path.join(EXPORT_DIR, f"{filename}.txt")
    if not os.path.exists(label_path):
        return JSONResponse(status_code=404, content={"error": "Labels not found"})

    with open(label_path) as f:
        labels = json.load(f)

    lines = []
    for obj in labels:
        x1, y1, x2, y2 = obj['bbox']
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

    with open(export_path, "w") as ef:
        ef.writelines(lines)

    return FileResponse(export_path, media_type="text/plain")

@app.post("/generate/")
async def generate_images(prompt: str = Form(...), count: int = Form(...)):
    return {"prompt": prompt, "images_requested": count}

# ------------------------------
# Celery Queue + Redis
# ------------------------------
from celery import Celery
celery_app = Celery('tasks', broker='redis://localhost:6379/0')

@celery_app.task
def render_synthetic_dataset(prompt: str, count: int):
    return f"Generated {count} images for prompt: {prompt}"

# ------------------------------
# PostgreSQL Schema
# ------------------------------
# Table: users
# - id (uuid, primary key)
# - email (string)
# - password_hash (string)
# - monthly_quota_used (int)

# Table: datasets
# - id
# - user_id (fk)
# - name
# - type (synthetic/manual)
# - created_at

# Table: images
# - id
# - dataset_id
# - filepath
# - label_json
# - generated_by (yolo/sam/manual)

# Storage: AWS S3 / Wasabi for image + label dumps

# ------------------------------
# Dockerfile (in root dir)
# ------------------------------
# FROM python:3.10-slim
# WORKDIR /app
# COPY . .
# RUN pip install -r requirements.txt
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

# ------------------------------
# docker-compose.yml (example)
# ------------------------------
# version: '3.8'
# services:
#   app:
#     build: .
#     ports:
#       - "8000:8000"
#     volumes:
#       - .:/app
#     depends_on:
#       - redis
#   redis:
#     image: redis:alpine
#     ports:
#       - "6379:6379"
