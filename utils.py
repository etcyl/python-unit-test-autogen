import subprocess
import json
from typing import Dict, Tuple
from tqdm import tqdm

def run_coverage(repo_path: str) -> Dict[str, float]:
    """
    Run pytest with coverage and return a dict of file coverage percentages.
    """
    try:
        subprocess.run(["pytest", "--cov", repo_path, "--cov-report", "json:coverage.json", "-q"], check=False)
        with open("coverage.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        return {k: v["percent_covered"] for k, v in data["files"].items()}
    except Exception:
        return {}

def progress_bar(iterable, desc: str):
    """Wrapper for tqdm progress display."""
    return tqdm(iterable, desc=desc, ncols=100, colour="green")
