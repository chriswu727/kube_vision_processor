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

@app.post("/upload")
async def upload_image(file: UploadFile, db: Session = Depends(get_db)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Store in Redis
        image_key = f"image:{file.filename}"
        redis_client.setex(
            image_key,
            3600,  # expire in 1 hour
            base64.b64encode(contents).decode()
        )
        
        # Store metadata in PostgreSQL
        db_image = models.Image(
            filename=file.filename,
            cache_key=image_key,
            format=image.format,
            size=f"{image.size[0]}x{image.size[1]}"
        )
        db.add(db_image)
        db.commit()
        
        result = {
            "filename": file.filename,
            "format": image.format,
            "size": image.size,
            "cache_key": image_key
        }
        
        return result
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

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