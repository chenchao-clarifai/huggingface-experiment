from typing import Dict

import datasets as hfd
import transformers as hf


def build_dataset_dict_from_config(cfg: Dict) -> hfd.DatasetDict:
    """
    Build HuggingFace DatastDict from Config Dict.

    Parameters
    ----------
    cfg : Dict
        Config Dict with fields:
        - splits: Dict
        - load_dataset: Dict
        - tokenizer: Dict
        - column_map: Dict
          - label_mapper: str
          - keep: List[str]: list of column names to keep

    Returns
    -------
    hfd.DatasetDict
        HuggingFace DatasetDict


    Example Config Yaml:
    ```yaml
    ---
    splits:
      train: train
      val: validation
      test: test
    load_dataset:
      path: amazon_reviews_multi
      name: en
    tokenizer:
      pretrained_model_name_or_path: bert-base-cased
    column_map:
      label_mapper: |
        lambda ex: \
        dict(labels=int(ex['stars'])-1,
            **(tokenizer('. '.join([ex['review_title'],
            ex['review_body']]))))
      keep:
        - labels
    ```
    """

    splits = cfg["splits"]
    # load
    load = cfg["load_dataset"]
    datadict = hfd.load_dataset(**load)
    # tokenizer
    tokenizer = hf.AutoTokenizer.from_pretrained(**cfg["tokenizer"])  # noqa
    # map
    label_mapper = eval(cfg["column_map"]["label_mapper"])
    keep_colns = cfg["column_map"]["keep"]
    assert isinstance(keep_colns, (list, tuple, set))
    rm_col = [
        col for col in datadict.column_names[splits["train"]] if col not in keep_colns
    ]
    datadict = datadict.map(label_mapper, remove_columns=rm_col)

    return datadict
