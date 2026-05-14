# ===== Imports ===== #
import torch

# ===== Global Configurations ===== # 
DATA_DIR = "./data"
MODEL_DIR = "./models"

BATCH_SIZE = 32
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-3
DROPOUT = 0.5

EPOCHS = 200
PATIENCE = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"