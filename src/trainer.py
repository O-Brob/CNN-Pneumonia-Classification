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
def model_eval(model: nn.Module, test_dl: DataLoader) -> None:
    """ 
    Evaluation loop for the Convolutional Neural Network, 
    given the provided test set data loader. 

    Args:
        model (nn.Module): The neural network architecture to evaluate.
        test_dl (DataLoader): The data loader providing (input, label) batches.
        
    Returns:
        None: The function outputs the evaluation accuracy on the standard output stream,
        alongside the fraction of correct predictions w.r.t. the total number of test set samples.
    """
    # Set model to evaluation mode
    model.eval()
    
    # Move the model to GPU if available
    print(f"Evaluation is initialized with (cuda/cpu): {config.DEVICE}")
    model.to(config.DEVICE)
    
    # Counters
    correct_preds = 0
    tot_samples   = 0
    
    print("Evaluation started. Please wait.")
    
    # Disable gradient calculation to save memory and speed up computations
    with torch.no_grad():
        for data in test_dl:
            inputs, labels = data
            
            # Move data to GPU if available
            inputs = inputs.to(config.DEVICE)
            labels = labels.to(config.DEVICE).float().unsqueeze(1)
            
            # Forward pass
            outputs = model(inputs)
            
            # Calculate accuracy.
            # Turn raw output logits into probabilities via sigmoid
            probs = torch.sigmoid(outputs) # sigmoid(x) = 1 / (1+e^(−x)​)
            preds = (probs >= 0.5).float() # .5 threshold to get pred: 0/1
            
            # Sum boolean representation of tensor and convert
            # to a standard numeric scalar for comparison
            correct_preds += (preds == labels).sum().item()
            tot_samples   += labels.size(0) # size(0) is batch size
            
    # Finalize accuracy percentage
    accuracy = (correct_preds / tot_samples) * 100
    
    # Output accuracy result
    print("Evaluation Completed:")
    print(f"Accuracy: {accuracy:.2f}%  --  ({correct_preds}/{tot_samples})")