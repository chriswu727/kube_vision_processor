import torch
from minio import Minio
import io
from facenet_pytorch import InceptionResnetV1
import torchvision.models as models
from torchvision import transforms
import numpy as np

class ModelManager:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"ModelManager initialized with device: {self.device}")
        
        if self.device.type == 'cuda':
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
            print(f"CUDA memory allocated: {torch.cuda.memory_allocated(0)/(1024**2):.2f} MB")
        
        self.minio_client = Minio(
            "minio:9000",
            access_key="minioadmin",
            secret_key="minioadmin",
            secure=False
        )
        self.face_model = None
        self.object_model = None
        
        # CIFAR-10 classes
        self.cifar_classes = [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]
        
        # Image transforms
        self.transform = transforms.Compose([
            transforms.Resize(32),  # Match CIFAR-10 size
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),  # CIFAR-10 values
                std=(0.2023, 0.1994, 0.2010)
            )
        ])

    def load_face_model(self):
        if self.face_model is None:
            print("Loading face recognition model...")
            # Get model from MinIO
            data = self.minio_client.get_object('models', 'face/inception_resnet_v1.pt')
            buffer = io.BytesIO(data.read())
            
            # Load state dict
            state_dict = torch.load(buffer, map_location=self.device)
            
            # Remove logits layers from state dict
            state_dict.pop('logits.weight', None)
            state_dict.pop('logits.bias', None)
            
            # Initialize model and load modified weights
            self.face_model = InceptionResnetV1(pretrained=None).to(self.device)
            self.face_model.load_state_dict(state_dict, strict=True)
            self.face_model.eval()
        return self.face_model

    def load_object_model(self):
        if self.object_model is None:
            print("Loading CIFAR-10 model...")
            # Initialize model first
            self.object_model = models.resnet18(pretrained=False)
            self.object_model.fc = torch.nn.Linear(512, len(self.cifar_classes))  # 10 classes
            
            # Get model from MinIO
            print("Getting model from MinIO...")
            data = self.minio_client.get_object('models', 'object/cifar10_resnet18.pt')
            buffer = io.BytesIO(data.read())
            
            # Load state dict
            print("Loading state dict...")
            state_dict = torch.load(buffer, map_location=self.device)
            
            # Load the complete state dict (including fc layer)
            self.object_model.load_state_dict(state_dict)
            
            # Move to device and set eval mode
            self.object_model.to(self.device)
            self.object_model.eval()
            print("CIFAR-10 model loaded successfully")
            
        return self.object_model

    def preprocess_image(self, image, image_type):
        if image_type == 'face':
            # Face images are already aligned by MTCNN
            img_tensor = self.transform(image).unsqueeze(0)
            return img_tensor.to(self.device)
        else:
            # Object images
            img_tensor = self.transform(image).unsqueeze(0)
            return img_tensor.to(self.device)

    def process_image(self, image, image_type):
        with torch.no_grad():
            if image_type == 'face':
                model = self.load_face_model()
                img_tensor = self.preprocess_image(image, 'face')
                embeddings = model(img_tensor)
                return {
                    'embeddings': embeddings.cpu().numpy().tolist(),
                    'type': 'face_embedding'
                }
            else:
                model = self.load_object_model()
                img_tensor = self.preprocess_image(image, 'object')
                
                # Get model output
                outputs = model(img_tensor)  # Shape: [1, num_classes]
                
                # Apply softmax directly to outputs
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                
                # Get top prediction from first (only) image
                prob, pred = torch.max(probabilities[0], dim=0)
                
                print(f"Raw output shape: {outputs.shape}")  # Debug print
                print(f"Predicted class: {self.cifar_classes[pred.item()]}")
                print(f"Confidence: {float(prob.item()):.4f}")
                
                return {
                    'class': self.cifar_classes[pred.item()],
                    'confidence': float(prob.item()),
                    'type': 'object_classification'
                } 