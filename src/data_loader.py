# ===== Imports ===== #
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from src import config

# ===== Method definitions ===== #

# TODO: !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# TODO: !!Does not use Augmentation to remove great biases yet!!
# TODO: !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

def create_dataloaders():
    # Create a standard transformation on all images s.t.
    # sizes are uniform and representation is as tensors.
    std_transform = transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize((256,256)),
        transforms.ToTensor()
    ])
    
    # Load the data directory
    dataset = datasets.ImageFolder(root=config.DATA_DIR, transform=std_transform)
    
    # Define fractional splits of data
    val_size   = int(0.15 * len(dataset))
    test_size  = int(0.15 * len(dataset))
    train_size = len(dataset) - val_size - test_size # ~70%
    
    # Perform splits
    train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])
    
    # Pass to DataLoader with shuffling on training data
    train_dl = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=config.BATCH_SIZE, shuffle=False)
    test_dl  = DataLoader(test_ds,  batch_size=config.BATCH_SIZE, shuffle=False)
    
    # Return the training, validation and testing dataloaders
    return (train_dl, val_dl, test_dl)