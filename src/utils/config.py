from typing import Dict

import yaml


def load_yaml(path_to_yaml: str) -> Dict:
    with open(path_to_yaml, "r") as f:
        return yaml.safe_load(f.read())
