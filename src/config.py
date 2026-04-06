# ===== Imports ===== #
import torch

# ===== Global Configurations ===== # 
DATA_DIR = "./data"
MODEL_DIR = "./models"

BATCH_SIZE = 32
LEARNING_RATE = 1e-4
DROPOUT = 0.2

EPOCHS = 20
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"