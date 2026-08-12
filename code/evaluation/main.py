import sys
import argparse
from pathlib import Path

# Add code dir to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.context import DatasetLoader
from checkpoints.manager import CheckpointManager
from evaluation.scorer import run_evaluation

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone Evaluation Runner")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use deterministic rules only, zero API calls",
    )
    parser.add_argument(
        "-y", "--yes",
        action="store_true",
        help="Bypass interactive confirmation prompts",
    )
    args = parser.parse_args()

    loader = DatasetLoader()
    checkpoint_mgr = CheckpointManager()
    run_evaluation(loader, checkpoint_mgr, yes=args.yes, dry_run=args.dry_run)
