from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    raw_path: str = "data/raw/reviews.csv"
    processed_dir: str = "data/processed"
    text_column: str = "text"
    label_column: str = "label"
    labels: list[str] = Field(default_factory=lambda: ["negative", "neutral", "positive"])
    max_length: int = 128
    train_ratio: float = 0.80
    val_ratio: float = 0.10
    test_ratio: float = 0.10
    use_class_weights: bool = True


class TeacherConfig(BaseModel):
    model_name: str = "aubmindlab/bert-base-arabertv02"
    epochs: int = 3
    learning_rate: float = 2.0e-5
    batch_size: int = 32
    weight_decay: float = 0.01
    warmup_ratio: float = 0.10
    mixed_precision: bool = True


class StudentConfig(BaseModel):
    num_layers: int = 6
    layers_from_teacher: list[int] = Field(default_factory=lambda: [0, 2, 4, 6, 8, 10])
    epochs: int = 5
    learning_rate: float = 5.0e-5
    batch_size: int = 32
    temperature: float = 2.0
    alpha: float = 0.50
    weight_decay: float = 0.01
    warmup_ratio: float = 0.10


class ExportConfig(BaseModel):
    artifacts_dir: str = "artifacts"
    onnx_fp32_path: str = "artifacts/student_fp32.onnx"
    onnx_int8_path: str = "artifacts/student_int8.onnx"
    tensorrt_engine_path: str = "artifacts/student_fp16.engine"
    benchmark_results_path: str = "artifacts/benchmark_results.csv"
    opset_version: int = 17


class TrackingConfig(BaseModel):
    experiment_name: str = "arabic-sentiment-distillation"
    tracking_uri: str = "mlruns"


class AppConfig(BaseModel):
    seed: int = 42
    data: DataConfig = Field(default_factory=DataConfig)
    teacher: TeacherConfig = Field(default_factory=TeacherConfig)
    student: StudentConfig = Field(default_factory=StudentConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)

    @classmethod
    def from_yaml(cls, path: str | Path = "configs/config.yaml") -> "AppConfig":
        config_path = Path(path)
        if not config_path.exists():
            return cls()
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(**data) if data else cls()
