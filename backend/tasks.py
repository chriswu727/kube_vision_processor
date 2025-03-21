from celery import Celery
import redis
import json
from sqlalchemy.orm import Session
import models

# Initialize Celery
celery_app = Celery('tasks',
    broker='amqp://user:password@rabbitmq:5672/',
    backend='redis://redis:6379/1'
)

redis_client = redis.Redis(host='redis', port=6379, db=0)

@celery_app.task
def process_image(cache_key: str, filename: str):
    try:
        # Get image from Redis
        image_data = redis_client.get(cache_key)
        if not image_data:
            raise Exception("Image not found in cache")
            
        # Simulate processing
        result = {
            "processed": True,
            "details": "Image processed successfully"
        }
        
        # Update database
        db = models.SessionLocal()
        db_image = db.query(models.Image).filter(models.Image.filename == filename).first()
        if db_image:
            db_image.status = "completed"
            db_image.result = result
            db.commit()
        
        return result
    except Exception as e:
        # Update database with error
        db = models.SessionLocal()
        db_image = db.query(models.Image).filter(models.Image.filename == filename).first()
        if db_image:
            db_image.status = "failed"
            db_image.result = {"error": str(e)}
            db.commit()
        raise
    finally:
        db.close() 