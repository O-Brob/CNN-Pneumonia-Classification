# ===== Imports ===== #
import torch
import getopt, sys
import os, shutil
import subprocess

from src import config
from src import data_loader
from src import model
from src import trainer

# ===== Fetch Args ===== #
args = sys.argv[1:] # User provided arguments
options = "hdtei:c"
long_options = ["help", "download", "train", "evaluate", "infer=", "clean"]

# ===== Main Method ===== #
def main():
    try:
        (arguments, _) = getopt.getopt(args, options, long_options)
        for (arg, _) in arguments:
            # ===== Output help message ===== #
            if arg in ("-h", "--help"):
                output =  "-h   --help            :   Print help information\n"
                output += "-d   --download        :   Download x-ray dataset for training\n"
                output += "-t   --train           :   Perform CNN training on GPU if available, else CPU\n"
                output += "-e   --evaluate        :   Evaluate performance after training over a test set\n"
                output += "-i   --infer [image]   :   Infer a classification on the given image \n"
                output += "-c   --clean           :   Remove downloaded datasets and saved models"
                print(output)
            
            # ===== Download dataset ===== #
            elif arg in ("-d", "--download"):
                print("Initializing dataset download. Please wait.")
                subprocess.run(
                    ["bash", "./curl_data.sh"], 
                    cwd=f"{config.DATA_DIR}/",
                    stdout=subprocess.DEVNULL
                )
                print("Dataset downloaded. Ready for training.")
                exit(0)
            
            # ===== Train CNN on dataset ===== #
            elif arg in ("-t", "--train"):
                # Check training datasets have been downloaded,
                # then execute training loop.
                if(not os.path.exists(f"{config.DATA_DIR}/Normal/") or not os.path.exists(f"{config.DATA_DIR}/Pneumonia/")):
                    print("Failed to initialize training: Dataset for training has not been downloaded.")
                    exit(1)
                # Load Data:
                (train, valid, _) = data_loader.create_dataloaders()
                
                # Initialize Model:
                cnn = model.CNN()
                
                optimizer = torch.optim.Adam(cnn.parameters(), config.LEARNING_RATE)
                criterion = torch.nn.BCEWithLogitsLoss()
                
                # Train Model:
                trainer.model_fit(cnn, optimizer, criterion, train, valid, config.PATIENCE)
            
            # ===== Evaluate performance of model on test data ===== #
            elif arg in ("-e", "--evaluate"):
                if(not os.path.exists(f"{config.MODEL_DIR}/trained_model.pt")):
                    print(f"Failed to evaluate model: Model not found in {config.MODEL_DIR}")
                    exit(1)
                
                # Load Data:
                (_, _, test) = data_loader.create_dataloaders()
                
                # Load Model:
                loaded_model = model.CNN()
                loaded_model.load_state_dict(torch.load(f"{config.MODEL_DIR}/trained_model.pt", weights_only=True))
                loaded_model.eval()
                
                # Evaluate Model:
                trainer.model_eval(loaded_model, test)
            
            # ===== Infer classification on input image ===== #
            elif arg in ("-i", "--infer"):
                # TODO: Fetch provided path (and check that provided path is valid and points to image)
                # TODO: Check that model exists.
                # TODO: Infer on i
                pass
            
            # ===== Clean downloaded dataset files and models ===== #
            elif arg in ("-c", "--clean"):
                normal_path = f"{config.DATA_DIR}/Normal"
                if os.path.exists(normal_path):
                    shutil.rmtree(normal_path)
                
                pneumonia_path = f"{config.DATA_DIR}/Pneumonia"
                if os.path.exists(pneumonia_path):
                    shutil.rmtree(pneumonia_path)
                
                model_path = f"{config.MODEL_DIR}/trained_model.pt"
                if os.path.exists(model_path):
                    os.remove(model_path)
                
                print("Cleanup complete!")
            
    except getopt.error as err:
        print(str(err)) # Report on exception

# ===== Boilerplate ===== #
if __name__ == "__main__":
    main()