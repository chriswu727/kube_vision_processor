from fastapi import FastAPI, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import redis
import json
import base64
from sqlalchemy.orm import Session
import models
from typing import List
from tasks import process_image
from facenet_pytorch import MTCNN
import torch

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Redis connection
redis_client = redis.Redis(host='redis', port=6379, db=0)

# Dependency for database session
def get_db():
    db = models.SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Add this at the start, after FastAPI initialization
@app.on_event("startup")
async def startup_event():
    models.init_db()

# Initialize MTCNN
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(device=device)

@app.post("/upload")
async def upload_image(file: UploadFile, db: Session = Depends(get_db)):
    try:
        # Read and convert image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Detect faces using MTCNN
        boxes, _ = mtcnn.detect(image)
        
        # Determine image type
        image_type = "face" if boxes is not None and len(boxes) > 0 else "object"
        
        # Store image in Redis
        image_key = f"image:{file.filename}"
        redis_client.setex(
            image_key,
            3600,  # 1 hour TTL
            base64.b64encode(contents).decode()
        )
        
        # Store metadata in PostgreSQL
        db_image = models.Image(
            filename=file.filename,
            cache_key=image_key,
            format=image.format,
            size=f"{image.size[0]}x{image.size[1]}",
            image_type=image_type,  # New field
            status="uploaded"
        )
        db.add(db_image)
        db.commit()
        
        return {
            "filename": file.filename,
            "cache_key": image_key,
            "image_type": image_type,
            "status": "uploaded"
        }
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process/{filename}")
async def process_image_request(filename: str, db: Session = Depends(get_db)):
    try:
        # Get image metadata
        db_image = db.query(models.Image).filter(models.Image.filename == filename).first()
        if not db_image:
            raise HTTPException(status_code=404, detail="Image not found")
        
        # Start celery task (update this line)
        task = process_image.apply_async(args=[db_image.cache_key, filename])
        
        return {"task_id": task.id, "status": "processing"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status/{filename}")
async def get_processing_status(filename: str, db: Session = Depends(get_db)):
    db_image = db.query(models.Image).filter(models.Image.filename == filename).first()
    if not db_image:
        raise HTTPException(status_code=404, detail="Image not found")
    
    if db_image.task_id:
        task = process_image.AsyncResult(db_image.task_id)
        return {
            "filename": filename,
            "status": db_image.status,
            "task_status": task.status,
            "result": task.result if task.ready() else None
        }
    
    return {"filename": filename, "status": db_image.status}

@app.get("/image/{filename}")
async def get_image(filename: str):
    image_key = f"image:{filename}"
    cached_image = redis_client.get(image_key)
    if not cached_image:
        raise HTTPException(status_code=404, detail="Image not found")
    return {"image": cached_image.decode()}

@app.get("/images", response_model=List[dict])
async def list_images(db: Session = Depends(get_db)):
    images = db.query(models.Image).all()
    return [{"filename": img.filename, "uploaded_at": img.uploaded_at} for img in images]

@app.get("/task/{task_id}")
async def get_task_status(task_id: str):
    task = process_image.AsyncResult(task_id)
    return {
        "task_id": task_id,
        "status": task.status,
        "result": task.result if task.ready() else None
    } 