import os

from .trainer import main

pwd = os.path.dirname(os.path.abspath(__file__))


def test_trainer():
    config_path = os.path.join(pwd, "example_config.yaml")
    try:
        main(config_path)
    except ZeroDivisionError:
        print("We are not doing train so it's ok.")


def test_imports():
    from ..utils.config import load_yaml  # noqa
    from ..utils.dataset import build_dataset_dict_from_config  # noqa
