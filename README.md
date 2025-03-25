# KubeVisionProcessor

A Kubernetes-based microservices application for image processing with GPU acceleration. The system is designed to handle image uploads, classify them as face or object images, and process them using different AI models dynamically.

## System Overview

This application demonstrates a modern microservices architecture where:
- Images are preprocessed to determine type (face/object)
- Images are temporarily cached for processing
- AI models are dynamically loaded from MinIO storage
- Tasks are distributed through message queues
- Processing status and results are tracked
- GPU resources are utilized efficiently

## Basic Workflow

The system processes images through a streamlined pipeline: When an image is uploaded, it's immediately preprocessed using MTCNN for face detection and temporarily stored in Redis cache. The preprocessing result (face/object) along with other metadata is recorded in PostgreSQL. When processing is requested, the image is queued in RabbitMQ for either FaceNet (for face embedding generation) or ResNet18 (for CIFAR-10 object classification) based on its recorded type. These AI models are dynamically loaded from MinIO storage by Celery workers, which utilize GPU acceleration for inference. The workers process images from the queue and store results back in PostgreSQL. Throughout this process, the system maintains efficient resource usage with automatic cleanup of expired cache entries and proper GPU resource allocation.

### Core Components

- **FastAPI Backend**: Handles API requests and GPU processing
  - Manages image uploads and preprocessing
  - Uses MTCNN for face detection
  - Labels images as 'face' or 'object'
  - Coordinates with other services
  - Utilizes GPU for processing tasks

- **MinIO Object Storage**: Model storage and management
  - Stores trained AI models
  - Enables dynamic model loading
  - Supports model versioning
  - Facilitates model updates without redeployment

- **Redis Cache**: Temporary storage system
  - Caches uploaded images (1-hour TTL)
  - Serves as Celery result backend
  - Optimizes data access

- **PostgreSQL**: Persistent metadata storage
  - Tracks image information and type
  - Stores processing status
  - Maintains task history
  - Auto-cleans expired image records

- **RabbitMQ & Celery**: Task queue system
  - RabbitMQ manages message distribution
  - Celery workers handle processing tasks
  - Dynamic model loading from MinIO
  - Ensures scalable task processing

### AI Models

1. Face Processing
   - Uses FaceNet (Inception-ResNet-V1)
   - Trained on VGGFace2 dataset
   - Generates face embeddings for future recognition

2. Object Classification
   - Uses ResNet18 architecture
   - Trained on CIFAR-10 dataset and saved to MinIO
   - Classifies 10 object categories:
     - airplane, automobile, bird, cat
     - deer, dog, frog, horse, ship, truck
   - GPU-accelerated training and inference

### API Endpoints

- `POST /upload` - Upload and preprocess image
- `POST /process/{filename}` - Queue processing task
- `GET /status/{filename}` - Check processing status
- `GET /image/{filename}` - Retrieve cached image
- `GET /images` - List all images
- `GET /task/{task_id}` - Get task details

## Next Development Phase

1. Storage Enhancement
   - Add permanent storage solution
   - Implement backup strategy
   - Optimize data flow

2. Frontend Development
   - Create user interface
   - Add real-time status updates
   - Display processing results

