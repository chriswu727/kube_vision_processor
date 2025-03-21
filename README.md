# KubeVisionProcessor

A Kubernetes-based microservices application for image processing with GPU acceleration. The system is designed to handle image uploads, process them using AI models, and manage the workflow through a distributed task queue.

## System Overview

This application demonstrates a modern microservices architecture where:
- Images are temporarily cached for processing
- Tasks are distributed through message queues
- Processing status and results are tracked
- GPU resources are utilized efficiently

### Core Components

- **FastAPI Backend**: Handles API requests and GPU processing
  - Manages image uploads and retrieval
  - Coordinates with other services
  - Utilizes GPU for processing tasks

- **Redis Cache**: Temporary storage system
  - Caches uploaded images (1-hour TTL)
  - Serves as Celery result backend
  - Optimizes data access

- **PostgreSQL**: Persistent metadata storage
  - Tracks image information
  - Stores processing status
  - Maintains task history

- **RabbitMQ & Celery**: Task queue system
  - RabbitMQ manages message distribution
  - Celery workers handle processing tasks
  - Ensures scalable task processing

### Current Implementation

The system currently supports:
1. Image upload and temporary storage
2. Task queuing and distribution
3. Status tracking and result retrieval
4. GPU resource allocation
5. Basic processing simulation

### API Endpoints

- `POST /upload` - Store new image
- `POST /process/{filename}` - Queue processing task
- `GET /status/{filename}` - Check processing status
- `GET /image/{filename}` - Retrieve cached image
- `GET /images` - List all images
- `GET /task/{task_id}` - Get task details

## Next Development Phase

1. AI Model Integration
   - Implement actual GPU processing
   - Add model inference pipeline
   - Optimize GPU utilization

2. Storage Enhancement
   - Add permanent storage solution
   - Implement backup strategy
   - Optimize data flow

3. Frontend Development
   - Create user interface
   - Add real-time status updates
   - Display processing results

