# ===== Imports ===== #
import json
import os
import torch

# ===== JSON Config Loader Class ===== #

class ConfigLoader:
    """
    The ConfigLoader class is used to assist in loading 
    configuration values from a config.json file.
    """
    def __init__(self, config_path : str):
        """The constructor of the ConfigLoader class.
        Given a config path, it ensures the config file exists,
        and loads the json content into a class instance variable.

        Args:
            config_path (str): The path to a JSON configuration file.
        """
        if(not os.path.exists(config_path)):
            print(f"Configuration file {config_path} does not exist.")
            exit(1)
        
        with open(config_path) as file:
            config = json.load(file)

        if(config["device"] == "auto"):
            config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Hold config as a state of this class
        self.config_dict = config
    
    def getValue(self, key : str):
        """ConfigLoader class method which, given a key, ensures
        that it exists and returns the value associated with the key.

        Args:
            key (str): The key to get associated value of.

        Returns:
            Any: The value associated with the provided key.
        """
        if (key not in self.config_dict):
            print(f"The key `{key}` does not exist in the config.")
            exit(1)

        return self.config_dict[key]

# ===== Global Configuration ===== #
configuration = ConfigLoader("./config.json")

# ===== Extract and expose configuration values ===== #
DATA_DIR  = configuration.getValue("data_dir")
MODEL_DIR = configuration.getValue("model_dir")

DATA_SPLIT_SEED = configuration.getValue("data_split_seed")

BATCH_SIZE    = configuration.getValue("batch_size")
LEARNING_RATE = configuration.getValue("learning_rate")
WEIGHT_DECAY  = configuration.getValue("weight_decay")
DROPOUT       = configuration.getValue("dropout")

EPOCHS   = configuration.getValue("epochs")
PATIENCE = configuration.getValue("patience")
DEVICE   = configuration.getValue("device")