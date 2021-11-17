from typing import Dict

import datasets as hfd
import torch
import transformers as hf

from ..utils.config import load_yaml
from ..utils.dataset import build_dataset_dict_from_config

DEVICE = "cuda:0"


def load_dataset(cfg: Dict) -> hfd.DatastDict:
    return build_dataset_dict_from_config(cfg)


def load_tokenizer(cfg: Dict) -> hf.AutoTokenizer:
    return hf.AutoTokenizer.from_pretrained(**cfg)


def load_model(cfg: Dict) -> torch.nn.Module:
    if "device" in cfg:
        device = cfg.pop("device")
    else:
        device = DEVICE
    model = hf.AutoModelForSequenceClassification.from_pretrained(**cfg)
    model.to(device)
    return model


def main(yaml_config_path: str) -> None:

    cfg = load_yaml(yaml_config_path)

    dataset_cfg = cfg["dataset_config"]
    model_cfg = cfg["model_config"]
    tokenizer_cfg = dataset_cfg["tokenizer"]
    trainer_cfg = cfg["trainer_config"]

    model = load_model(model_cfg)
    tokenizer = load_tokenizer(tokenizer_cfg)

    dataset_dict = load_dataset(dataset_cfg)
    if "train" in dataset_dict:
        train_dataset = dataset_dict["train"]
    else:
        train_dataset = None

    if "eval" in dataset_dict:
        eval_dataset = dataset_dict["eval"]
    else:
        eval_dataset = None

    training_args = hf.TrainingArguments(**trainer_cfg)
    trainer = hf.Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_config_path", type=str, nargs=None)
    kwargs = vars(parser.parse_args())

    main(**kwargs)
