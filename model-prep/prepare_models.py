import torch
from facenet_pytorch import InceptionResnetV1
import torchvision.models as models
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from minio import Minio
import io

def train_cifar10_model():
    print("Starting CIFAR-10 model training...")
    
    # Optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    
    # Enhanced data transforms
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # Load CIFAR-10 dataset
    print("Loading CIFAR-10 dataset...")
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                          download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, 
                                            batch_size=128,  # Back to 128
                                            shuffle=True,
                                            num_workers=4,   # Use 4 workers
                                            pin_memory=True) # Enable pin memory

    # Initialize model
    print("Initializing ResNet18 with ImageNet weights...")
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)
    
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    model = model.to(device)
    print("Model moved to device successfully")
    
    # Training parameters
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    
    # Training loop
    print("Training for 50 epochs...")
    best_acc = 0
    for epoch in range(50):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if i % 50 == 49:
                avg_loss = running_loss / 50
                accuracy = 100 * correct / total
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {avg_loss:.3f}, accuracy: {accuracy:.2f}%')
                running_loss = 0.0
                correct = 0
                total = 0
        
        scheduler.step()
        print(f'Epoch {epoch + 1} completed')
    
    print("Finished Training")
    return model

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

    # 1. Download and save FaceNet model (commented out as it's working fine)
    """
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
    """

    # 2. Train and save CIFAR-10 model
    print("Training CIFAR-10 model...")
    cifar_model = train_cifar10_model()
    cifar_model.eval()
    
    # Save trained model
    print("Saving and uploading trained CIFAR-10 model...")
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