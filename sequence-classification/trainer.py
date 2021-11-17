import datasets as hfd
import torch
import transformers as hf

from ..utils.config import load_yaml
from ..utils.dataset import build_dataset_dict_from_config

DEVICE = "cuda:0"


def load_dataset(path_to_yaml: str) -> hfd.DatastDict:
    cfg = load_yaml(path_to_yaml)
    return build_dataset_dict_from_config(cfg)


def load_model(path_to_yaml: str) -> torch.nn.Module:
    cfg = load_yaml(path_to_yaml)
    if "device" in cfg:
        device = cfg.pop("device")
    else:
        device = DEVICE
    model = hf.AutoModelForSequenceClassification.from_pretrained(**cfg)
    model.to(device)
    return model
