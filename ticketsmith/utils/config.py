import yaml
import hashlib
import json
import os

def load_config(config_path):
    """Loads a YAML config file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def hash_config(config):
    """Returns a SHA256 hash of the config dictionary."""
    config_str = json.dumps(config, sort_keys=True)
    return hashlib.sha256(config_str.encode('utf-8')).hexdigest()
