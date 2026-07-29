import os
import json 
DB_PATH = "/mnt/truenas_db/user/christina"

TRAIN_PARAMS = {
    "gh_original": {
        "cnn_batch_size": 32,
        "cnn_epochs": 5,
        "cnn_lr": 10e-3,
        "seq_batch_size": 32,
        "seq_epochs": 15,
    },
    "paper_original": {
        "cnn_batch_size": 128,
        "cnn_epochs": 10,
        "cnn_lr": 0.001,
        "seq_batch_size": 32,
        "seq_epochs": 15,
    }
}

class Configuration:
    """
    Class for MorpheusNet training configuration

    Attributes:
        base_config_path (str): Path to the base configuration file.
        override_config_path (str): Path to the override configuration file (optional).

    Methods:
        _load_config(base_config_path, override_config_path): Loads the configuration from the specified files.
        _update_config(base_config, override_config): Updates the base configuration with values from the override configuration.
        get(key, default=None): Retrieves a value from the configuration using the specified key.
    """

    def __init__(self, base_config_path, override_config_path=None, name=None):
        """
        Initializes the Configuration object.

        Args:
            base_config_path (str): Path to the base configuration file.
            override_config_path (str): Path to the override configuration file (optional).
            name (str): Name of the experiment (optional).

        Raises:
            FileNotFoundError: If the configuration file does not exist.
            ValueError: If the configuration file is not in a valid format.
        """

        self.config = self._load_config(base_config_path, override_config_path)
        self.dataset = self.config.get("dataset", {})
        self.training = self.config.get("training_params", {})
        self.name = self.config.get("name", os.path.basename(override_config_path).replace(".json", "") if override_config_path else os.path.basename(base_config_path).replace(".json", ""))
        if name is not None:
            self.name = name
        # default to paper-specified hyperparams since they give the best results
        self.run_type = self.config.get("run_type", "paper_original")
        self.hparams = TRAIN_PARAMS.get(self.run_type, "paper_original")
        self.data_path = self.dataset.get("path", "")
        if os.path.isabs(self.data_path):
            self.data_path = self.data_path
        else:
            self.data_path = os.path.join(DB_PATH, self.data_path) 

        self.window_length = self.training.get("epoch_duration", 30)  * 100
        self.save_dir = os.path.join(DB_PATH,'morpheus', self.name)

    def _load_config(self, base_config_path, override_config_path):
        """
        Loads the configuration from the specified files.

        Args:
            base_config_path (str): Path to the base configuration file.
            override_config_path (str): Path to the override configuration file (optional).

        Returns:
            dict: The loaded configuration.
        """
        with open(base_config_path) as base_config_file:
            base_config = json.load(base_config_file)
        # override if necessary
        if override_config_path:
            with open(override_config_path) as override_config_file:
                override_config = json.load(override_config_file)
            
            # update any values if they are found in the override, else keep the ones from base
            end_config = self._update_config(base_config, override_config)
        return end_config if override_config_path else base_config
    
    def _update_config(self, base_config, override_config):
        for key, value in override_config.items():
            if key in base_config and isinstance(base_config[key], dict) and isinstance(value, dict):
                self._update_config(base_config[key], value)
            else:
                base_config[key] = value
        return base_config

    def get(self, key, default=None):
        """
        Retrieves a value from the configuration using the specified key."""
        return self.config.get(key, default)