import os

import torch
import transformers as hf

from ..utils.config import load_yaml
from .trainer import load_dataset, load_model, load_tokenizer

LOCAL_RANK = int(os.environ["LOCAL_RANK"])
assert LOCAL_RANK >= 0

DEVICE = f"cuda:{LOCAL_RANK}" if torch.cuda.is_available() else "cpu"


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

    if "test" in dataset_dict:
        eval_dataset = dataset_dict["test"]
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
    if LOCAL_RANK == 0:
        trainer.save_model()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_config_path", type=str, nargs=None)
    kwargs = vars(parser.parse_args())

    main(**kwargs)
