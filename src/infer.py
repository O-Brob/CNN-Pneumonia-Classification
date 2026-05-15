# ===== Imports ===== #
import torch
import os

from src import config
from torch import nn
from torchvision import transforms
from PIL import Image

# ===== Inference methods ===== #

def model_infer(model: nn.Module, image_path : str) -> None:
    """
    Performs inference on an image using the provided model.
    Converts the output after forward passing into a probability,
    and makes a prediction for the inferred image. 
    The prediction is returned on the standard stream
    alongside a confidence level for the prediction.

    Args:
        model (nn.Module): The model which will be used for inference.
        image_path (str): The path to the image file to be classified.
    """
    # Set model to evaluation mode
    model.eval()
    
    # Move the model to GPU if available
    print(f"Inference is initialized with (cuda/cpu): {config.DEVICE}")
    model.to(config.DEVICE)
    
    image_tensor = _process_image(image_path)
    
    with torch.no_grad():
        # Forward pass through model
        output = model(image_tensor)
        
        # Convert output logits to probability via sigmoid
        probability = torch.sigmoid(output).item()
        
        # Get prediction via .5 threshold
        pred = 1 if probability >= 0.5 else 0
        
        # Flip confidence level if prediction is "healthy",
        # to represent confidence that it is NOT pneumonia.
        confidence = probability if pred == 1 else 1 - probability
    
    # Output results on standard output stream
    print(f"\nInference Completed:")
    print("=============================================================================")
    print(f"File: {os.path.basename(image_path)}")
    print(f"Prediction: {"Pneumonia" if pred == 1 else "Healthy"}")
    print(f"Confidence: {(confidence * 100):.2f}%")
    print("=============================================================================")

# ===== Helper methods ===== #

def _process_image(image_path : str) -> torch.Tensor:
    """
    Loads the provided image and processes it using the same transforms
    as during model training. Converts the result to a tensor of the
    model's expected input format and passes it to the configured device. 

    Args:
        image_path (str): The path of the image to process.

    Returns:
        torch.Tensor: A Tensor object of the image, ready for inference.
    """
    # Load the image via path
    img = Image.open(image_path)
    
    # Apply same transforms as used during training in dataloader.py.
    std_transform = transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        
        # Z-Score Normalization, mean & std calculated
        # for this dataset specifically
        transforms.Normalize(mean=[0.4815], std=[0.2364])
    ])
    
    # Transformed image, convert to 
    # tensor and pass to selected device
    img_tensor = torch.as_tensor(std_transform(img))
    img_tensor = img_tensor.unsqueeze(0)
    img_tensor = img_tensor.to(config.DEVICE)
    
    return img_tensor