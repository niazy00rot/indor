import torch
import torch.nn as nn
from torchvision import models, transforms

CLASSES = [
    'airport_inside', 'artstudio', 'auditorium', 'bakery', 'bar',
    'bathroom', 'bedroom', 'bookstore', 'bowling', 'buffet',
    'casino', 'children_room', 'church_inside', 'classroom', 'cloister',
    'closet', 'clothingstore', 'computerroom', 'concert_hall', 'corridor'
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
model_info = {}
def load_model(model_path=f"DL_Final_Project/best_resnet50_model.pth"):
    global model, model_info

    checkpoint = torch.load(model_path, map_location=device)

    model = models.resnet50(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(num_features, 1024),
        nn.ReLU(),
        nn.BatchNorm1d(1024),
        nn.Dropout(0.4),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(0.3),
        nn.Linear(512, len(CLASSES))
    )

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    model_info = {
        'device': str(device),
        'classes': len(CLASSES)
    }


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def predict_image(image):
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
    
    top5_prob, top5_idx = torch.topk(probabilities, 5)
    
    predictions = []
    for i in range(5):
        predictions.append({
            'class': CLASSES[top5_idx[0][i].item()],
            'class_name': CLASSES[top5_idx[0][i].item()].replace('_', ' ').title(),
            'confidence': float(top5_prob[0][i].item() * 100)
        })
    
    return predictions
