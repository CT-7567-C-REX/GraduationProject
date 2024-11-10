import os
import secrets
from flaskWebSite import app

def save_picture(form_picture, wheretosave):
    randomhex = secrets.token_hex(8)
    _, f_ext = os.path.splitext(form_picture.filename)
    picture_fn = randomhex + f_ext
    path = os.path.join(app.root_path, wheretosave, picture_fn)
    form_picture.save(path)
    return picture_fn

from torch import nn
import torch
# Create a convolutional neural network 
class FashionMNISTModelV2(nn.Module):
    """
    Model architecture copying TinyVGG from: 
    https://poloclub.github.io/cnn-explainer/
    """
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, 
                      out_channels=hidden_units, 
                      kernel_size=3, # how big is the square that's going over the image?
                      stride=1, # default
                      padding=1),# options = "valid" (no padding) or "same" (output has same shape as input) or int for specific number 
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, 
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2) # default stride value is same as kernel_size
        )
        self.block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # Where did this in_features shape come from? 
            # It's because each layer of our network compresses and changes the shape of our input data.
            nn.Linear(in_features=hidden_units * 16 * 16, 
                      out_features=output_shape)
        )
    
    def forward(self, x: torch.Tensor):
        x = self.block_1(x)
        # print(x.shape)
        x = self.block_2(x)
        # print(x.shape)
        x = self.classifier(x)
        # print(x.shape)
        return x

if torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

loaded_model_2 = FashionMNISTModelV2(input_shape=3, 
                                    hidden_units=10,
                                    output_shape=20)

 # Load in the saved state_dict()
loaded_model_2.load_state_dict(torch.load(f="project_env/flaskWebSite/models/03_pytorch_computer_vision_model_2.pth"))

# Send model to GPU
loaded_model_2 = loaded_model_2.to(device)

import torchvision.transforms as transforms
transform = transforms.Compose(
    [transforms.Resize((64, 64)),
     transforms.ToTensor(),
     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

classes = ['Not sure',
 'T-Shirt',
 'Shoes',
 'Shorts',
 'Shirt',
 'Pants',
 'Skirt',
 'Other',
 'Top',
 'Outwear',
 'Dress',
 'Body',
 'Longsleeve',
 'Undershirt',
 'Hat',
 'Polo',
 'Blouse',
 'Hoodie',
 'Skip',
 'Blazer']

from PIL import Image
def predict_single_image(image_path ,model=loaded_model_2, device: torch.device = device):
    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    image = transform(image)  # Apply the transformation
    image = image.unsqueeze(0).to(device)  # Add batch dimension and move to device
    
    # Make prediction
    model.eval()
    with torch.inference_mode():
        pred_logits = model(image)  # Get raw model outputs (logits)
        pred_prob = torch.softmax(pred_logits.squeeze(), dim=0)  # Convert logits to probabilities
    
    pred_prob = torch.argmax(pred_prob).item()
    
    return classes[pred_prob] 