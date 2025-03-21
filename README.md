# KubeVisionProcessor

A Kubernetes-based microservices application for image processing with GPU acceleration. The system is designed to handle image uploads, classify them as face or object images, and process them using different AI models.

## System Overview

This application demonstrates a modern microservices architecture where:
- Images are preprocessed to determine type (face/object)
- Images are temporarily cached for processing
- Tasks are distributed through message queues
- Processing status and results are tracked
- GPU resources are utilized efficiently

### Core Components

- **FastAPI Backend**: Handles API requests and GPU processing
  - Manages image uploads and preprocessing
  - Uses MTCNN for face detection
  - Labels images as 'face' or 'object'
  - Coordinates with other services
  - Utilizes GPU for processing tasks

- **Redis Cache**: Temporary storage system
  - Caches uploaded images (1-hour TTL)
  - Serves as Celery result backend
  - Optimizes data access

- **PostgreSQL**: Persistent metadata storage
  - Tracks image information and type
  - Stores processing status
  - Maintains task history

- **RabbitMQ & Celery**: Task queue system
  - RabbitMQ manages message distribution
  - Celery workers handle processing tasks
  - Ensures scalable task processing

### Current Implementation

The system currently supports:
1. Image upload and preprocessing
   - Face detection using MTCNN
   - Image type labeling (face/object)
2. Temporary image storage in Redis
3. Metadata storage in PostgreSQL
4. Task queuing and distribution
5. Status tracking and result retrieval
6. GPU resource allocation

### API Endpoints

- `POST /upload` - Upload and preprocess image
- `POST /process/{filename}` - Queue processing task
- `GET /status/{filename}` - Check processing status
- `GET /image/{filename}` - Retrieve cached image
- `GET /images` - List all images
- `GET /task/{task_id}` - Get task details

## Next Development Phase

1. AI Model Integration
   - Implement face recognition for face images
   - Implement CIFAR-10 classification for object images
   - Dynamic model loading based on image type
   - Optimize GPU utilization

2. Storage Enhancement
   - Add permanent storage solution
   - Implement backup strategy
   - Optimize data flow

3. Frontend Development
   - Create user interface
   - Add real-time status updates
   - Display processing results

