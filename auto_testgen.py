import os
import sys
import yaml
import subprocess
from typing import Tuple
from logger import setup_logger
from llm_client import LLMClient
from test_generator import UnitTestGenerator


def run_pytest(self, test_file: str) -> Tuple[bool, str]:
    """Execute pytest on a test file and return (passed, output)."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", test_file],
            capture_output=True,
            text=True,
            env=os.environ  # ensures same Conda env
        )
        passed = result.returncode == 0
        return passed, result.stdout + result.stderr
    except Exception as e:
        self.logger.exception(f"Error running pytest on {test_file}: {e}")
        return False, str(e)


def main():
    """Entrypoint for the automated test generation system."""
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    logger = setup_logger("auto_testgen", cfg.get("log_level", "INFO"))
    logger.info("=== Auto TestGen Started ===")

    try:
        llm = LLMClient(
            model_path=cfg["model_path"],
            temperature=cfg["temperature"],
            max_tokens=cfg["max_tokens"],
            context_window=cfg["context_window"],
            logger=logger
        )
        gen = UnitTestGenerator(
            llm=llm,
            output_dir=cfg["output_dir"],
            max_fix_attempts=cfg["max_fix_attempts"],
            metadata_file=cfg["metadata_file"],
            coverage_threshold=cfg["coverage_threshold"],
            logger=logger
        )
        gen.process_repository(cfg["repo_path"])
    except Exception as e:
        logger.exception(f"Fatal error: {e}")

    logger.info("=== Auto TestGen Completed ===")


if __name__ == "__main__":
    main()
