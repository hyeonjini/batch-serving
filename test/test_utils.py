def test_load_config():
    from src.utils import load_yaml
    import os

    config = load_yaml("./config.yaml", attr=False)
    text_classification_config = config["text_classification"]
    text_classification_model_name = text_classification_config["model"]["name"]
    text_classification_model_path = text_classification_config["model"]["path"]

    assert text_classification_model_name == "ElectraForSequenceClassification"
    assert text_classification_model_path == "./misc/model_repository/food_name_classification_pretrained_model"

def test_load_config_attr():
    from src.utils import load_yaml
    import addict

    config = load_yaml("./config.yaml")
    text_cls_config = config.text_classification

    model_name = text_cls_config.model.name
    model_path = text_cls_config.model.path

    tokenizer_path = text_cls_config.tokenizer.path
    tokenizer_config = text_cls_config.tokenizer.config

    assert isinstance(config, addict.Dict)
    assert model_name == "ElectraForSequenceClassification"
    assert model_path == "./misc/model_repository/food_name_classification_pretrained_model"
    assert tokenizer_path == "monologg/koelectra-base-v3-discriminator"
    assert isinstance(tokenizer_config, dict)

def test_load_image():
    pass