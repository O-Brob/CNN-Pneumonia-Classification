# ===== Imports ===== #
import torch
from src import config
from torch import nn, optim
from torch.utils.data import DataLoader

# ===== Training methods ===== #

# Fit the model to the training data
def model_fit(
    model: nn.Module, 
    optimizer: optim.Optimizer, 
    criterion: nn.modules.loss._Loss, 
    train_dl: DataLoader,
    val_dl: DataLoader,
    patience: int) -> None:
    """Training loop for the Convolutional Neural Network.

    Args:
        model (nn.Module): The neural network architecture to train.
        optimizer (optim.Optimizer): The optimization algorithm.
        criterion (nn.modules.loss._Loss): The loss function.
        train_dl (DataLoader): The data loader providing (input, label) batches.
        
    Returns:
        None: The function saves the model to disk and prints progress to the console.
    """
    # Move the model to GPU if available
    print(f"Training is initialized with (cuda/cpu): {config.DEVICE}")
    model.to(config.DEVICE)
    
    # Variables to track holdout state
    lowest_val_loss  = float("inf")
    patience_counter = 0
    
    for epoch in range(config.EPOCHS):
        # ===== Training Phase ===== #
        model.train()
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
            fit_loss += loss.item()
            
            # Backpropagate (calculate gradient anew)
            loss.backward()
            
            # Update weights of models based on backward propagation
            optimizer.step()
            
        # Get average training loss for this epoch
        avg_fit_loss = fit_loss / len(train_dl)
            
        # ===== Validation Phase ===== #
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for data in val_dl:
                val_inputs, val_labels = data
                
                #  Move data to GPU if available
                val_inputs = val_inputs.to(config.DEVICE)
                val_labels = val_labels.to(config.DEVICE).float().unsqueeze(1)
                
                # Forward pas to get initial predictions
                val_outputs = model(val_inputs)
                
                # Loss calculation using given criterion
                val_batch_loss = criterion(val_outputs, val_labels)
                
                # Add loss to val_loss 
                val_loss += val_batch_loss.item()
            
        # Get average validation loss for this epoch
        avg_val_loss = val_loss /len(val_dl)
        
        print(f"[Epoch: {epoch + 1}] Train Loss: {avg_fit_loss:.5f}  --  Val Loss: {avg_val_loss:.5f}")
        
        # ===== Holdout Check ===== #
        # New best on validation set
        # --> Reset patience since we saw improvement
        # --> Save new lowest loss and the new best model
        if(avg_val_loss < lowest_val_loss):
            patience_counter = 0
            lowest_val_loss = avg_val_loss
            torch.save(model.state_dict(), config.MODEL_DIR + "/trained_model.pt")
        
        # No improvement, increase patience counter for holdout validation
        else:
            patience_counter += 1
            
        # If maximum patience has been met, quit training. 
        if(patience_counter >= patience):
            print("Training ended due to early stopping.")
            break
    
    # Notify that training has been completed
    print("Training Complete")

# ===== Evaluation Methods ===== #

# Evaluate the model's performance over an unseen data set
def model_eval(model: nn.Module, eval_dl: DataLoader) -> None:
    """ 
    Evaluation loop for the Convolutional Neural Network, given the
    provided data loader for a data set to evaluate model performance on. 
    Evaluation is done via standard accuracy calculations, 
    as well as context recall, precision, false positive rate and F1-Score.
    For more info on context recall, precision, false positive rate and F1-Score, see:
    https://developers.google.com/machine-learning/crash-course/classification/accuracy-precision-recall

    Args:
        model (nn.Module): The neural network architecture to evaluate.
        eval_dl (DataLoader): The data loader providing (input, label) batches to evaluate on.
        
    Returns:
        None: The function outputs the evaluation accuracy, context recall, precision, false positive rate and F1-Score on 
        the standard output stream, alongside the fraction of correct predictions w.r.t. the total number of eval set samples.
        The Confusion Matrix of the evaluation is also output.
    """
    # Set model to evaluation mode
    model.eval()
    
    # Move the model to GPU if available
    print(f"Evaluation is initialized with (cuda/cpu): {config.DEVICE}")
    model.to(config.DEVICE)
    
    # Counters for Accuracy
    correct_preds = 0
    tot_samples   = 0
    
    # Counters for Confusion Matrix and evaluation metrics
    true_pos   = 0
    true_negs  = 0
    
    false_pos  = 0
    false_negs = 0
    
    print("Evaluation started. Please wait.")
    
    # Disable gradient calculation to save memory and speed up computations
    with torch.no_grad():
        for data in eval_dl:
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
            
            # == Counter Updates == #
            # Sum boolean representation of tensor and convert
            # to a standard numeric scalar for comparison
            correct_preds += (preds == labels).sum().item()
            tot_samples   += labels.size(0) # size(0) is batch size
            
            true_pos   += ((preds == 1) & (labels == 1)).sum().item()
            true_negs  += ((preds == 0) & (labels == 0)).sum().item()
            false_pos  += ((preds == 1) & (labels == 0)).sum().item()
            false_negs += ((preds == 0) & (labels == 1)).sum().item()
            
    # Finalize accuracy percentage
    accuracy = (correct_preds / tot_samples) * 100 if tot_samples > 0 else 0.0
    
    # Recall = TP / (TP + FN)
    actual_positives = true_pos + false_negs
    recall = (true_pos / actual_positives) * 100 if actual_positives > 0 else 0.0
    
    # Precision = TP / (TP + FP)
    all_classed_positives = true_pos + false_pos
    precision = (true_pos / all_classed_positives) * 100 if all_classed_positives > 0 else 0.0
    
    # False Positive Rate = FP / (FP + TN)
    all_actual_negatives = false_pos + true_negs
    fpr = (false_pos / all_actual_negatives) * 100 if all_actual_negatives > 0 else 0.0
    
    # F1 Score = 2TP / (2TP + FP + FN) = 2 * ((precision * recall) / (precision + recall))
    f1_divider = (precision + recall)
    f1_score = 2 * ((precision * recall) / f1_divider) if f1_divider > 0 else 0.0
    
    # Output accuracy result
    print("\nEvaluation Completed:")
    print("=============================================================================")
    print(f"Accuracy: {accuracy:.2f}%  --  ({correct_preds}/{tot_samples})")
    print(f"Recall: {recall:.2f}% -- ({true_pos}/{actual_positives} pneumonia cases caught)")
    print(f"Precision: {precision:.2f}% -- ({true_pos}/{all_classed_positives} pneumonia predictions were correct)")
    print(f"False Positive Rate: {fpr:.2f}% -- ({false_pos}/{all_actual_negatives} healthy incorrectly flagged as pneumonia)")
    print(f"F1 Score: {f1_score:.2f}%") # Harmonic mean of Recall & Precision
    print("=============================================================================")
    print()
    print("Confusion Matrix:")
    print(f"TP: {true_pos}\t| FP: {false_pos}")
    print(f"FN: {false_negs}\t| TN: {true_negs}")