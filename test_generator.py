import os
import ast
import json
import time
import subprocess
import logging
import sys
import shutil
from typing import List, Dict, Tuple
from tqdm import tqdm

from utils import run_coverage, progress_bar


class UnitTestGenerator:
    """
    Handles scanning source files, generating pytest code,
    executing tests, applying self-fixes, and tracking coverage improvements.
    Streams output to file and self-chains continuation if cutoff detected.
    """

    def __init__(self, llm: "LLMClient", output_dir: str, max_fix_attempts: int,
                 metadata_file: str, coverage_threshold: int, logger: logging.Logger):
        self.llm = llm
        self.output_dir = output_dir
        self.max_fix_attempts = max_fix_attempts
        self.metadata_file = metadata_file
        self.coverage_threshold = coverage_threshold
        self.logger = logger
        os.makedirs(self.output_dir, exist_ok=True)
        self.meta = self._load_metadata()

    # -------------------------------------------------------------------------
    # Metadata handling
    # -------------------------------------------------------------------------
    def _load_metadata(self) -> Dict[str, Dict]:
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_metadata(self) -> None:
        with open(self.metadata_file, "w", encoding="utf-8") as f:
            json.dump(self.meta, f, indent=2)

    # -------------------------------------------------------------------------
    # Source scanning
    # -------------------------------------------------------------------------
    def extract_functions(self, filepath: str) -> List[str]:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                tree = ast.parse(f.read())
            return [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
        except Exception as e:
            self.logger.warning(f"AST parse failed for {filepath}: {e}")
            return []

    # -------------------------------------------------------------------------
    # Streaming + chaining generation
    # -------------------------------------------------------------------------
    def generate_tests_for_file(self, filepath: str, attempt: int = 0) -> str:
        """
        Streams model output to file and automatically continues if cutoff detected.
        Uses ast.parse() to verify syntax completeness.
        """
        funcs = self.extract_functions(filepath)
        if not funcs:
            self.logger.info(f"No functions detected in {filepath}. Skipping.")
            return ""

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                code = f.read()

            base_prompt = f"""
You are an expert QA engineer.
Generate complete, valid pytest unit tests that verify correctness and edge cases for these functions:
{funcs}

Rules:
- Output only valid Python code (no markdown fences or commentary).
- Ensure code is syntactically valid and not truncated.
- Include realistic edge cases and clear assertion messages.
- End with a newline after the final test.

Source code:
{code}
"""

            file_name = os.path.basename(filepath)
            test_file_name = f"test_{file_name}"
            test_file_path = os.path.join(self.output_dir, test_file_name)
            if attempt > 0:
                test_file_path = f"{test_file_path}.retry{attempt}"
            tmp_path = test_file_path + ".tmp"

            os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
            self.logger.info(f"Streaming test generation to {tmp_path}")

            def stream_to_file(prompt, append=False):
                """Stream response to file with live tqdm and flush."""
                mode = "a" if append else "w"
                with open(tmp_path, mode, encoding="utf-8") as out_f, tqdm(
                    desc=f"Generating {test_file_name}" + (" (continued)" if append else ""),
                    unit="tok",
                    dynamic_ncols=True,
                    colour="green"
                ) as pbar:
                    for out in self.llm.llm.create_completion(
                        prompt=prompt,
                        temperature=self.llm.temperature,
                        max_tokens=self.llm.max_tokens,
                        stream=True
                    ):
                        if "choices" in out and out["choices"]:
                            chunk = out["choices"][0].get("text", "")
                            if chunk:
                                out_f.write(chunk)
                                out_f.flush()
                                pbar.update(len(chunk.split()))
                                print(chunk, end="", flush=True)

            # Initial generation
            stream_to_file(base_prompt)

            # Retry self-chaining if file is incomplete
            for i in range(3):
                if self._is_code_complete(tmp_path):
                    break
                self.logger.warning(f"⚠️ Detected incomplete code. Continuing generation (pass {i+1})...")
                continuation_prompt = (
                    "Continue generating from where you left off. "
                    "Do not repeat previous code. Output only valid Python."
                )
                stream_to_file(continuation_prompt, append=True)
                time.sleep(0.5)

            if not self._is_code_complete(tmp_path):
                self.logger.error(f"❌ Still incomplete after retries: {test_file_name}")

            shutil.move(tmp_path, test_file_path)
            self.logger.info(f"\n✅ Completed writing {test_file_path}")
            return test_file_path

        except Exception as e:
            self.logger.exception(f"Streaming generation error for {filepath}: {e}")
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            return ""

    # -------------------------------------------------------------------------
    # Completeness detection
    # -------------------------------------------------------------------------
    def _is_code_complete(self, path: str) -> bool:
        """Check for syntactically valid Python via ast.parse()."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                code = f.read().strip()
            if not code:
                return False
            ast.parse(code)
            # Ensure file ends properly (no unterminated blocks)
            last_line = code.splitlines()[-1]
            if last_line.strip().startswith(("def ", "class ")) or last_line.strip().endswith(":"):
                return False
            return True
        except SyntaxError as e:
            self.logger.warning(f"⚠️ Incomplete or invalid syntax detected ({e}).")
            return False
        except Exception as e:
            self.logger.warning(f"⚠️ Unexpected parse error: {e}")
            return False

    # -------------------------------------------------------------------------
    # Pytest runner
    # -------------------------------------------------------------------------
    def run_pytest(self, test_file: str) -> Tuple[bool, str]:
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", "-q", test_file],
                capture_output=True,
                text=True,
                env=os.environ
            )
            return result.returncode == 0, result.stdout + result.stderr
        except Exception as e:
            self.logger.exception(f"Error running pytest on {test_file}: {e}")
            return False, str(e)

    # -------------------------------------------------------------------------
    # Fix attempts
    # -------------------------------------------------------------------------
    def attempt_fixes(self, test_file: str, source_file: str) -> None:
        """Attempt iterative LLM repairs on failing tests."""
        for attempt in range(1, self.max_fix_attempts + 1):
            passed, output = self.run_pytest(test_file)
            if passed:
                self.logger.info(f"✅ {test_file} passed on attempt {attempt}.")
                return
            self.logger.warning(f"Attempt {attempt} failed for {test_file}. Retrying...")
            fix_prompt = f"""
Fix the following pytest failures while preserving test intent.
Include valid Python code only.

Source file: {source_file}
Test file: {test_file}
Pytest output:
{output}
"""
            self.generate_tests_for_file(source_file, attempt=attempt)
            time.sleep(1)
        self.logger.error(f"❌ Max fix attempts reached for {test_file}")

    # -------------------------------------------------------------------------
    # Repository loop
    # -------------------------------------------------------------------------
    def process_repository(self, repo_path: str) -> None:
        coverage = run_coverage(repo_path)
        all_files = [
            os.path.join(r, f)
            for r, _, fs in os.walk(repo_path)
            for f in fs if f.endswith(".py") and not f.startswith("test_")
        ]
        for filepath in progress_bar(all_files, desc="Analyzing files"):
            coverage_score = coverage.get(filepath, 0)
            file_name = os.path.basename(filepath)

            if coverage_score >= self.coverage_threshold:
                self.logger.info(f"{file_name} already meets coverage ({coverage_score:.1f}%). Skipping.")
                continue

            test_file_path = self.generate_tests_for_file(filepath)
            if not test_file_path:
                continue

            self.attempt_fixes(test_file_path, filepath)
            new_cov = run_coverage(repo_path).get(filepath, coverage_score)
            self.meta[file_name] = {
                "previous_coverage": coverage_score,
                "new_coverage": new_cov
            }
            self._save_metadata()

        self._print_summary()

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    def _print_summary(self):
        self.logger.info("\n=== Summary Report ===")
        for f, stats in self.meta.items():
            prev, new = stats["previous_coverage"], stats["new_coverage"]
            delta = new - prev
            self.logger.info(f"{f:25s} {prev:6.1f}% → {new:6.1f}%  ({delta:+.1f}%)")
import os
import ast
import json
import time
import subprocess
import logging
import sys
import shutil
from typing import List, Dict, Tuple
from tqdm import tqdm

from utils import run_coverage, progress_bar


class UnitTestGenerator:
    """
    Handles scanning source files, generating pytest code,
    executing tests, applying self-fixes, and tracking coverage improvements.
    Streams output to file and self-chains continuation if cutoff detected.
    """

    def __init__(self, llm: "LLMClient", output_dir: str, max_fix_attempts: int,
                 metadata_file: str, coverage_threshold: int, logger: logging.Logger):
        self.llm = llm
        self.output_dir = output_dir
        self.max_fix_attempts = max_fix_attempts
        self.metadata_file = metadata_file
        self.coverage_threshold = coverage_threshold
        self.logger = logger
        os.makedirs(self.output_dir, exist_ok=True)
        self.meta = self._load_metadata()

    # -------------------------------------------------------------------------
    # Metadata handling
    # -------------------------------------------------------------------------
    def _load_metadata(self) -> Dict[str, Dict]:
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_metadata(self) -> None:
        with open(self.metadata_file, "w", encoding="utf-8") as f:
            json.dump(self.meta, f, indent=2)

    # -------------------------------------------------------------------------
    # Source scanning
    # -------------------------------------------------------------------------
    def extract_functions(self, filepath: str) -> List[str]:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                tree = ast.parse(f.read())
            return [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
        except Exception as e:
            self.logger.warning(f"AST parse failed for {filepath}: {e}")
            return []

    # -------------------------------------------------------------------------
    # Streaming + chaining generation
    # -------------------------------------------------------------------------
    def generate_tests_for_file(self, filepath: str, attempt: int = 0) -> str:
        """
        Streams model output to file and automatically continues if cutoff detected.
        Uses ast.parse() to verify syntax completeness.
        """
        funcs = self.extract_functions(filepath)
        if not funcs:
            self.logger.info(f"No functions detected in {filepath}. Skipping.")
            return ""

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                code = f.read()

            base_prompt = f"""
You are an expert QA engineer.
Generate complete, valid pytest unit tests that verify correctness and edge cases for these functions:
{funcs}

Rules:
- Output only valid Python code (no markdown fences or commentary).
- Ensure code is syntactically valid and not truncated.
- Include realistic edge cases and clear assertion messages.
- End with a newline after the final test.

Source code:
{code}
"""

            file_name = os.path.basename(filepath)
            test_file_name = f"test_{file_name}"
            test_file_path = os.path.join(self.output_dir, test_file_name)
            if attempt > 0:
                test_file_path = f"{test_file_path}.retry{attempt}"
            tmp_path = test_file_path + ".tmp"

            os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
            self.logger.info(f"Streaming test generation to {tmp_path}")

            def stream_to_file(prompt, append=False):
                """Stream response to file with live tqdm and flush."""
                mode = "a" if append else "w"
                with open(tmp_path, mode, encoding="utf-8") as out_f, tqdm(
                    desc=f"Generating {test_file_name}" + (" (continued)" if append else ""),
                    unit="tok",
                    dynamic_ncols=True,
                    colour="green"
                ) as pbar:
                    for out in self.llm.llm.create_completion(
                        prompt=prompt,
                        temperature=self.llm.temperature,
                        max_tokens=self.llm.max_tokens,
                        stream=True
                    ):
                        if "choices" in out and out["choices"]:
                            chunk = out["choices"][0].get("text", "")
                            if chunk:
                                out_f.write(chunk)
                                out_f.flush()
                                pbar.update(len(chunk.split()))
                                print(chunk, end="", flush=True)

            # Initial generation
            stream_to_file(base_prompt)

            # Retry self-chaining if file is incomplete
            for i in range(3):
                if self._is_code_complete(tmp_path):
                    break
                self.logger.warning(f"⚠️ Detected incomplete code. Continuing generation (pass {i+1})...")
                continuation_prompt = (
                    "Continue generating from where you left off. "
                    "Do not repeat previous code. Output only valid Python."
                )
                stream_to_file(continuation_prompt, append=True)
                time.sleep(0.5)

            if not self._is_code_complete(tmp_path):
                self.logger.error(f"❌ Still incomplete after retries: {test_file_name}")

            shutil.move(tmp_path, test_file_path)
            self.logger.info(f"\n✅ Completed writing {test_file_path}")
            return test_file_path

        except Exception as e:
            self.logger.exception(f"Streaming generation error for {filepath}: {e}")
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            return ""

    # -------------------------------------------------------------------------
    # Completeness detection
    # -------------------------------------------------------------------------
    def _is_code_complete(self, path: str) -> bool:
        """Check for syntactically valid Python via ast.parse()."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                code = f.read().strip()
            if not code:
                return False
            ast.parse(code)
            # Ensure file ends properly (no unterminated blocks)
            last_line = code.splitlines()[-1]
            if last_line.strip().startswith(("def ", "class ")) or last_line.strip().endswith(":"):
                return False
            return True
        except SyntaxError as e:
            self.logger.warning(f"⚠️ Incomplete or invalid syntax detected ({e}).")
            return False
        except Exception as e:
            self.logger.warning(f"⚠️ Unexpected parse error: {e}")
            return False

    # -------------------------------------------------------------------------
    # Pytest runner
    # -------------------------------------------------------------------------
    def run_pytest(self, test_file: str) -> Tuple[bool, str]:
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", "-q", test_file],
                capture_output=True,
                text=True,
                env=os.environ
            )
            return result.returncode == 0, result.stdout + result.stderr
        except Exception as e:
            self.logger.exception(f"Error running pytest on {test_file}: {e}")
            return False, str(e)

    # -------------------------------------------------------------------------
    # Fix attempts
    # -------------------------------------------------------------------------
    def attempt_fixes(self, test_file: str, source_file: str) -> None:
        """Attempt iterative LLM repairs on failing tests."""
        for attempt in range(1, self.max_fix_attempts + 1):
            passed, output = self.run_pytest(test_file)
            if passed:
                self.logger.info(f"✅ {test_file} passed on attempt {attempt}.")
                return
            self.logger.warning(f"Attempt {attempt} failed for {test_file}. Retrying...")
            fix_prompt = f"""
Fix the following pytest failures while preserving test intent.
Include valid Python code only.

Source file: {source_file}
Test file: {test_file}
Pytest output:
{output}
"""
            self.generate_tests_for_file(source_file, attempt=attempt)
            time.sleep(1)
        self.logger.error(f"❌ Max fix attempts reached for {test_file}")

    # -------------------------------------------------------------------------
    # Repository loop
    # -------------------------------------------------------------------------
    def process_repository(self, repo_path: str) -> None:
        coverage = run_coverage(repo_path)
        all_files = [
            os.path.join(r, f)
            for r, _, fs in os.walk(repo_path)
            for f in fs if f.endswith(".py") and not f.startswith("test_")
        ]
        for filepath in progress_bar(all_files, desc="Analyzing files"):
            coverage_score = coverage.get(filepath, 0)
            file_name = os.path.basename(filepath)

            if coverage_score >= self.coverage_threshold:
                self.logger.info(f"{file_name} already meets coverage ({coverage_score:.1f}%). Skipping.")
                continue

            test_file_path = self.generate_tests_for_file(filepath)
            if not test_file_path:
                continue

            self.attempt_fixes(test_file_path, filepath)
            new_cov = run_coverage(repo_path).get(filepath, coverage_score)
            self.meta[file_name] = {
                "previous_coverage": coverage_score,
                "new_coverage": new_cov
            }
            self._save_metadata()

        self._print_summary()

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    def _print_summary(self):
        self.logger.info("\n=== Summary Report ===")
        for f, stats in self.meta.items():
            prev, new = stats["previous_coverage"], stats["new_coverage"]
            delta = new - prev
            self.logger.info(f"{f:25s} {prev:6.1f}% → {new:6.1f}%  ({delta:+.1f}%)")
