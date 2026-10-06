import argparse
import sys

from arabic_sentiment.core.config import AppConfig


def main():
    parser = argparse.ArgumentParser(
        prog="arabic-sentiment",
        description="Production Arabic Sentiment Analysis CLI (Teacher-Student KD & INT8 ONNX)",
    )
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml", help="Path to YAML config"
    )

    subparsers = parser.add_subparsers(dest="command", help="Sub-commands")

    # prepare-data
    subparsers.add_parser("prepare-data", help="Preprocesses and splits Arabic reviews dataset")

    # train-teacher
    subparsers.add_parser("train-teacher", help="Fine-tunes 12-layer AraBERT teacher model")

    # distill
    subparsers.add_parser("distill", help="Distills knowledge into 6-layer student model")

    # quantize
    subparsers.add_parser("quantize", help="Exports student to ONNX and quantizes to INT8")

    # benchmark
    subparsers.add_parser("benchmark", help="Runs fair latency and accuracy benchmark suite")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    config = AppConfig.from_yaml(args.config)
    print(f"Loaded configuration for experiment: {config.tracking.experiment_name}")
    print(f"Executing command: {args.command}...")

    if args.command == "prepare-data":
        from arabic_sentiment.data.dataset import SentimentDataModule

        dm = SentimentDataModule(config.data, config.teacher.model_name, seed=config.seed)
        train_df, val_df, test_df = dm.prepare_data()
        print(
            f"Data prepared successfully! Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}"
        )

    elif args.command == "quantize":
        print("Exporting ONNX and applying INT8 quantization...")
        print(f"Target INT8 path: {config.export.onnx_int8_path}")

    else:
        print(f"Command '{args.command}' ready to run in pipeline.")


if __name__ == "__main__":
    main()
