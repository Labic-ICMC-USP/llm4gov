import argparse
import logging
from .icl_classifier import ICLClassifier

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ICL-based text classifier using an LLM and YAML configuration."
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="config.yaml",
        help="Path to the YAML configuration file (default: config.yaml).",
    )
    args = parser.parse_args()

    logger.info("Running icl-classifier CLI", extra={"config_path": args.config})
    classifier = ICLClassifier(config_path=args.config)
    results = classifier.run()
    classifier.save_results(results)
    logger.info("icl-classifier CLI finished successfully")


if __name__ == "__main__":
    main()
