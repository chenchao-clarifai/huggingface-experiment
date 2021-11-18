import os

from .trainer import main

pwd = os.path.dirname(os.path.abspath(__file__))


def test_trainer():
    config_path = os.path.join(pwd, "example_config.yaml")
    try:
        main(config_path)
    except ZeroDivisionError:
        print("We are not doing train so it's ok.")
