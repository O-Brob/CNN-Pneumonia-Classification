# ===== Imports ===== #
import torch
from src import config
from torch import nn, optim
from torch.utils.data import DataLoader

# ===== Training methods ===== #

# TODO: !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# TODO: !!Does not use holdout validation yet!!
# TODO: !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# Fit the model to the training data
def model_fit(
    model: nn.Module, 
    optimizer: optim.Optimizer, 
    criterion: nn.modules.loss._Loss, 
    train_dl: DataLoader) -> None:
    """Training loop for the Convolutional Neural Network.

    Args:
        model (nn.Module): The neural network architecture to train.
        optimizer (optim.Optimizer): The optimization algorithm.
        criterion (nn.modules.loss._Loss): The loss function.
        train_dl (DataLoader): The data loader providing (input, label) batches.
        
    Returns:
        None: The function saves the model to disk and prints progress to the console.
    """
    # Set model in training mode
    model.train()
    
    # Move the model to GPU if available
    print(f"Training is initialized with (cuda/cpu): {config.DEVICE}")
    model.to(config.DEVICE)
    
    # TODO: Replace with holdout validation
    for epoch in range(config.EPOCHS):
        fit_loss = 0.0
        
        # Enumarate to track progress
        for i, data in enumerate(train_dl, 0):
            inputs, labels = data
            
            # Move data to GPU if available
            inputs = inputs.to(config.DEVICE)
            labels = labels.to(config.DEVICE).float().unsqueeze(1)
            
            # Clear gradients of optimized tensors,
            # to avoid gradient accumulation, prevent memory leaks,
            # and ensure gradient calc is "freshly calculated" for each batch
            optimizer.zero_grad()
            
            # Forward pass to get initial predictions
            outputs = model(inputs)
            
            # Loss calculation using given criterion
            loss = criterion(outputs, labels)
            
            # Backpropagate (calculate gradient anew)
            loss.backward()
            
            # Update weights of models based on backward propagation
            optimizer.step()
            
            # Output progress
            fit_loss += loss.item()
            if(i % 100 == 99):
                print(f"[Epoch: {epoch +1}, {i + 1:5d}] Loss: {fit_loss / 100}")
                fit_loss = 0.0
    
    # Save and finish training
    torch.save(model.state_dict(), config.MODEL_DIR + "/trained_model.pt")
    print("Training Complete")

# ===== Evaluation Methods ===== #

# Evaluate the model's performance over an unseen test set
def model_eval(model, test_dl):
    pass