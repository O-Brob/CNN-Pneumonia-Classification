# ===== Imports ===== #
import torch
import getopt, sys
import os, shutil
import subprocess
from PIL import Image

from src import config
from src import data_loader
from src import model
from src import trainer
from src import infer

# ===== Fetch Args ===== #
args = sys.argv[1:] # User provided arguments
options = "hdtei:c"
long_options = ["help", "download", "train", "evaluate", "infer=", "clean"]

# ===== Main Method ===== #
def main():
    try:
        (optlist, _) = getopt.getopt(args, options, long_options)
        for (opt, arg) in optlist:
            # ===== Output help message ===== #
            if opt in ("-h", "--help"):
                output =  "-h   --help            :   Print help information\n"
                output += "-d   --download        :   Download x-ray dataset for training\n"
                output += "-t   --train           :   Perform CNN training on GPU if available, else CPU\n"
                output += "-e   --evaluate        :   Evaluate performance after training over a test set\n"
                output += "-i   --infer [image]   :   Infer a classification on the given image \n"
                output += "-c   --clean           :   Remove downloaded datasets and saved models"
                print(output)
            
            # ===== Download dataset ===== #
            elif opt in ("-d", "--download"):
                print("Initializing dataset download. Please wait.")
                subprocess.run(
                    ["bash", "./curl_data.sh"], 
                    cwd=f"{config.DATA_DIR}/",
                    stdout=subprocess.DEVNULL
                )
                print("Dataset downloaded. Ready for training.")
            
            # ===== Train CNN on dataset ===== #
            elif opt in ("-t", "--train"):
                # Check training datasets have been downloaded,
                # then execute training loop.
                if(not os.path.exists(f"{config.DATA_DIR}/Normal/") or not os.path.exists(f"{config.DATA_DIR}/Pneumonia/")):
                    print("Failed to initialize training: Dataset for training has not been downloaded.")
                    exit(1)
                # Load Data:
                (train, valid, _) = data_loader.create_dataloaders(config.DATA_DIR)
                
                # Initialize Model:
                cnn = model.CNN()
                
                optimizer = torch.optim.Adam(
                    cnn.parameters(), 
                    config.LEARNING_RATE, 
                    weight_decay=config.WEIGHT_DECAY)
                criterion = torch.nn.BCEWithLogitsLoss()
                
                # Train Model:
                trainer.model_fit(cnn, optimizer, criterion, train, valid, config.PATIENCE)
            
            # ===== Evaluate performance of model on test data ===== #
            elif opt in ("-e", "--evaluate"):
                # Ensure the trained model exists
                if(not os.path.exists(f"{config.MODEL_DIR}/trained_model.pt")):
                    print(f"Failed to evaluate model: Model not found in {config.MODEL_DIR}")
                    exit(1)
                
                # Load Data:
                (_, _, test) = data_loader.create_dataloaders(config.DATA_DIR)
                
                # Load Model:
                loaded_model = model.CNN()
                loaded_model.load_state_dict(torch.load(f"{config.MODEL_DIR}/trained_model.pt", weights_only=True))
                
                # Evaluate Model:
                trainer.model_eval(loaded_model, test)
            
            # ===== Infer classification on input image ===== #
            elif opt in ("-i", "--infer"):
                # Get the absolute path of the file
                abs_path = os.path.abspath(arg)
                
                # Ensure the trained model exists
                if(not os.path.exists(f"{config.MODEL_DIR}/trained_model.pt")):
                    print(f"Failed to infer: Model not found in {config.MODEL_DIR}")
                    exit(1)
                
                # Ensure the provided file exists
                if(not os.path.isfile(abs_path)):
                    print(f"Failed to infer: Could not find the file")
                    exit(1)
                
                # If file exists, ensure it is indeed an image file.
                try: 
                    with Image.open(abs_path) as img: 
                        img.verify()
                except:
                    print(f"Failed to infer: The file is not an image file")
                    exit(1)
                
                # Load Model:
                loaded_model = model.CNN()
                loaded_model.load_state_dict(torch.load(f"{config.MODEL_DIR}/trained_model.pt", weights_only=True))
                
                # Infer classification on the image passed as argument
                infer.model_infer(loaded_model, abs_path)
            
            # ===== Clean downloaded dataset files and models ===== #
            elif opt in ("-c", "--clean"):
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