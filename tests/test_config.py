from arabic_sentiment.core.config import AppConfig


def test_app_config_defaults():
    config = AppConfig()
    assert config.seed == 42
    assert config.data.max_length == 128
    assert config.teacher.epochs == 3
    assert config.student.num_layers == 6
    assert config.student.layers_from_teacher == [0, 2, 4, 6, 8, 10]
    assert config.export.opset_version == 17


def test_app_config_from_yaml():
    config = AppConfig.from_yaml("configs/config.yaml")
    assert config.seed == 42
    assert config.data.raw_path == "data/raw/reviews.csv"
    assert config.teacher.model_name == "aubmindlab/bert-base-arabertv02"
