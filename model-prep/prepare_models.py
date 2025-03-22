import torch
from facenet_pytorch import InceptionResnetV1
import torchvision.models as models
from minio import Minio
import io

def prepare_and_upload_models():
    # Initialize MinIO client
    minio_client = Minio(
        "minio:9000",
        access_key="minioadmin",
        secret_key="minioadmin",
        secure=False
    )

    # Create buckets if they don't exist
    if not minio_client.bucket_exists("models"):
        minio_client.make_bucket("models")

    # 1. Download and save FaceNet model
    print("Downloading FaceNet model...")
    facenet = InceptionResnetV1(pretrained='vggface2')
    
    buffer = io.BytesIO()
    torch.save(facenet.state_dict(), buffer)
    buffer.seek(0)
    
    print("Uploading FaceNet model to MinIO...")
    minio_client.put_object(
        "models",
        "face/inception_resnet_v1.pt",
        buffer,
        buffer.getbuffer().nbytes
    )

    # 2. Download and save CIFAR-10 model
    print("Downloading CIFAR-10 model...")
    cifar_model = models.resnet18(pretrained=True)
    
    buffer = io.BytesIO()
    torch.save(cifar_model.state_dict(), buffer)
    buffer.seek(0)
    
    print("Uploading CIFAR-10 model to MinIO...")
    minio_client.put_object(
        "models",
        "object/cifar10_resnet18.pt",
        buffer,
        buffer.getbuffer().nbytes
    )
    
    print("All models prepared and uploaded successfully!")

if __name__ == "__main__":
    prepare_and_upload_models() 