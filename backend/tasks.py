from celery import Celery
import redis
import json
from sqlalchemy.orm import Session
import models
from PIL import Image as PILImage
import io
import base64
from model_manager import ModelManager
import torch
import os
import multiprocessing as mp

# Set multiprocessing start method to 'spawn'
if mp.get_start_method(allow_none=True) is None:
    mp.set_start_method('spawn')

# Initialize Celery
celery_app = Celery('tasks',
    broker='amqp://user:password@rabbitmq:5672/',
    backend='redis://redis:6379/1'
)

# Configure Celery to use spawn
celery_app.conf.update(
    worker_max_tasks_per_child=1,
    worker_prefetch_multiplier=1,
    worker_pool='solo'        # Use solo pool instead of prefork
)

# Initialize Redis
redis_client = redis.Redis(host='redis', port=6379, db=0)

# Initialize ModelManager
model_manager = ModelManager()
print(f"Celery worker GPU availability: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Celery worker CUDA device: {torch.cuda.get_device_name(0)}")

@celery_app.task(bind=True)
def process_image(self, cache_key: str, filename: str):
    print(f"Processing {filename} on device: {model_manager.device}")
    try:
        # Get database session
        db = Session(models.engine)
        
        # Get image metadata from database
        db_image = db.query(models.Image).filter(models.Image.filename == filename).first()
        if not db_image:
            raise ValueError(f"Image {filename} not found in database")
        
        # Update status to processing
        db_image.status = "processing"
        db_image.task_id = self.request.id
        db.commit()
        
        try:
            # Get image from Redis
            image_data = redis_client.get(cache_key)
            if not image_data:
                raise ValueError(f"Image {filename} not found in Redis cache")
            
            # Convert to PIL Image
            image_bytes = base64.b64decode(image_data)
            image = PILImage.open(io.BytesIO(image_bytes))
            
            # Process image based on type
            result = model_manager.process_image(image, db_image.image_type)
            
            # Update database with results
            db_image.status = "completed"
            db_image.result = result
            if db_image.image_type == 'object':
                db_image.confidence = result.get('confidence')
            db.commit()
            
            return {
                "status": "success",
                "filename": filename,
                "result": result
            }
            
        except Exception as e:
            # Update status to failed
            db_image.status = "failed"
            db.commit()
            raise e
            
    except Exception as e:
        return {
            "status": "error",
            "filename": filename,
            "error": str(e)
        }
        
    finally:
        db.close() 