# ===== Imports ===== #
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from collections import defaultdict
import random
import re

from src import config

# ===== Method definitions ===== #

def _extract_patient_id(filename: str, label: int) -> str:
    """
    Helper method to extract unique patient identifiers from filenames.
    
    Pneumonia format: 
      - person{id}_{type}_{image#}.jpeg
    Normal formats: 
      - IM-{id}-{image#}.jpeg
      - IM-{id}-{image#}-{sequence}.jpeg
      - NORMAL2-IM-{id}-{image#}.jpeg
      - NORMAL2-IM-{id}-{image#}-{sequence}.jpeg
    
    Args:
        filename (str): The image filename
        label (int): 0 for Normal, 1 for Pneumonia
    
    Returns:
        str: Unique patient identifier
    """
    if label == 1:  # Pneumonia
        # Extract "person{id}_{type}" from "person{id}_{type}_{image#}.jpeg"
        match = re.match(r"(person\d+_(?:bacteria|virus))", filename)
        if match:
            return match.group(1)

    else:  # Normal
        # Extract patient ID from IM-{id} and include source prefix to distinguish between
        # IM-*.jpeg and NORMAL2-IM-*.jpeg files as IDs seem to be reused for different patients.
        match = re.search(r"IM-(\d+)", filename)
        if match:
            if filename.startswith("NORMAL2"):
                return f"normal_NORMAL2_IM{match.group(1)}"
            else:
                return f"normal_IM{match.group(1)}"

    # Fallback
    return filename

def create_dataloaders(data_dir : str) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates three DataLoaders for Training, Validation, and Testing with a 70-15-15 split.
    Preprocessing is done on index level rather than holding images in memory for efficiency.
    Undersamples the dominant dataset to match the size of the smaller dataset to remove
    weighted biases in the training data without resorting to augmentation, e.g. in
    cases where flipping images creates unrealistic features for training data, as in for X-Rays.

    Args:
        data_dir (str): The root of a directory containing folders of binary classified data

    Returns:
        tuple[DataLoader, DataLoader, DataLoader]: Train, Validation and Testing DataLoaders
    """
    # Create a standard transformation on all images s.t.
    # sizes are uniform and representation is as tensors.
    std_transform = transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        
        # Z-Score Normalization, mean & std calculated
        # for this dataset specifically
        transforms.Normalize(mean=[0.4815], std=[0.2364])
    ])
    
    # Load the data directory
    dataset = datasets.ImageFolder(root=data_dir, transform=std_transform)
    
    # Extract labels and group indices by patient IDs and class
    target_labels      = dataset.targets
    
    # defaultdicts to automatically add lists for new patients
    normal_patients    = defaultdict(list)
    pneumonia_patients = defaultdict(list)
    
    # Extract patient ids and group indices to ids:
    # e.g.
    # pneumonia_patients = {
    #   "person1_bacteria": [100, 105, 110],
    #   "person1_virus": [101, 102],
    #   ...
    # }
    for i, label in enumerate(target_labels):
        filename = dataset.imgs[i][0].split("/")[-1]
        patient_id = _extract_patient_id(filename, label)
        
        if label == 0:
            normal_patients[patient_id].append(i)
        else:
            pneumonia_patients[patient_id].append(i)
    
    # Get list of unique patients per class
    normal_patient_ids    = list(normal_patients.keys())
    pneumonia_patient_ids = list(pneumonia_patients.keys())
    
    # Undersample to balance classes at a patient level
    min_patients = min(len(normal_patient_ids), len(pneumonia_patient_ids))
    normal_patient_ids    = random.sample(normal_patient_ids, min_patients)
    pneumonia_patient_ids = random.sample(pneumonia_patient_ids, min_patients)
    
    # Combine the patients and shuffle
    all_patient_ids = normal_patient_ids + pneumonia_patient_ids
    random.shuffle(all_patient_ids)
    
    # Split patients (not imgs!!) into train/val/test sets.
    # Done to prevent patient leakage between datasets.
    total_patients = len(all_patient_ids)
    val_patients   = int(0.15 * total_patients)
    test_patients  = int(0.15 * total_patients)
    train_patients = total_patients - val_patients - test_patients
    
    train_patient_ids = set(all_patient_ids[:train_patients])
    val_patient_ids   = set(all_patient_ids[train_patients:train_patients + val_patients])
    # (Test Patient IDs are inferred through not being in train/val)
    
    # Gather indices for each split
    train_idxs, val_idxs, test_idxs = [], [], []
    
    # Extend lists with indices for images of 
    # patients assigned to that data set.
    for id in normal_patient_ids:
        if id in train_patient_ids:
            train_idxs.extend(normal_patients[id])
        elif id in val_patient_ids:
            val_idxs.extend(normal_patients[id])
        else:
            test_idxs.extend(normal_patients[id])
    
    for id in pneumonia_patient_ids:
        if id in train_patient_ids:
            train_idxs.extend(pneumonia_patients[id])
        elif id in val_patient_ids:
            val_idxs.extend(pneumonia_patients[id])
        else:
            test_idxs.extend(pneumonia_patients[id])
    
    # Create subsets using the indexes gathered above
    train_ds = Subset(dataset, train_idxs)
    val_ds   = Subset(dataset, val_idxs)
    test_ds  = Subset(dataset, test_idxs)
    
    # Pass to DataLoader with shuffling on training data
    train_dl = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=config.BATCH_SIZE, shuffle=False)
    test_dl  = DataLoader(test_ds,  batch_size=config.BATCH_SIZE, shuffle=False)
    
    # Return the training, validation and testing dataloaders
    return (train_dl, val_dl, test_dl)