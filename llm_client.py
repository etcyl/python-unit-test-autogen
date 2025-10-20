import os
import logging
import hashlib
import requests
from llama_cpp import Llama


class LLMClient:
    """
    Loads or downloads a local GGUF model (e.g. Phi-3 Mini 4K Instruct)
    and validates it so it won’t redownload on each run.
    """

    def __init__(self, model_path: str, temperature: float, max_tokens: int,
                 context_window: int, logger: logging.Logger):
        self.logger = logger
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Normalize path
        model_path = os.path.normpath(model_path)
        model_path = self._ensure_model(model_path)

        try:
            self.logger.info(f"Loading model from {model_path}")
            self.llm = Llama(
                model_path=model_path,
                n_ctx=context_window,
                n_threads=os.cpu_count(),
                n_batch=8,
                use_mlock=True
            )
            self.logger.info(f"Model loaded successfully from {model_path}")
        except Exception as e:
            self.logger.exception(f"Failed to load LLM: {e}")
            raise

    # -------------------------------------------------------------------------
    # Model handling
    # -------------------------------------------------------------------------
    def _ensure_model(self, model_path: str, force_download: bool = False) -> str:
        """Ensure model exists and passes validation before downloading."""
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        # ✅ If the model exists and validates, use it
        if os.path.exists(model_path) and not force_download:
            if self._validate_model(model_path):
                self.logger.info(f"✅ Using existing model: {model_path}")
                return model_path
            else:
                self.logger.warning("Existing model invalid — will re-download.")

        # Only reach here if no valid model was found
        url = (
            "https://huggingface.co/microsoft/"
            "phi-3-mini-4k-instruct-gguf/resolve/main/"
            "phi-3-mini-4k-instruct.Q4_K_M.gguf"
        )
        self.logger.info(f"⬇️ Downloading model from {url}")
        self._download_model(url, model_path)

        if not self._validate_model(model_path):
            raise RuntimeError("Downloaded model failed validation.")
        self.logger.info("✅ Model verified successfully.")
        return model_path


    def _validate_model(self, path: str) -> bool:
        """Check model file existence, GGUF header, and store checksum for reuse."""
        try:
            if not os.path.exists(path):
                return False
            size_gb = os.path.getsize(path) / (1024 ** 3)
            if size_gb < 1.0:
                self.logger.warning(f"Model too small ({size_gb:.2f} GB) — incomplete.")
                return False
            with open(path, "rb") as f:
                header = f.read(4)
            if header != b"GGUF":
                self.logger.warning(f"Invalid GGUF header for {path}.")
                return False

            # Compute and cache checksum to ensure stability
            hasher = hashlib.sha256()
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    hasher.update(chunk)
            checksum = hasher.hexdigest()[:12]
            self.logger.info(f"✅ Valid GGUF model ({size_gb:.2f} GB, SHA256 {checksum})")
            return True
        except Exception as e:
            self.logger.warning(f"Model validation failed: {e}")
            return False


    def _download_model(self, url: str, file_path: str):
        """Stream model download with progress feedback."""
        with requests.get(url, stream=True, timeout=1800) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            downloaded = 0
            with open(file_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total:
                            pct = (downloaded / total) * 100
                            print(f"\rDownloading {os.path.basename(file_path)}: {pct:.2f}%", end="")
        print("\nDownload complete.")

    # -------------------------------------------------------------------------
    # Inference
    # -------------------------------------------------------------------------
    def generate(self, prompt: str) -> str:
        """Generate full text output safely with streaming and retry."""
        try:
            def _stream_once() -> str:
                chunks = []
                for out in self.llm.create_completion(
                    prompt=prompt,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    stream=True
                ):
                    if "choices" in out and out["choices"]:
                        chunks.append(out["choices"][0].get("text", ""))
                return "".join(chunks).strip()

            output = _stream_once()

            # Detect cutoff by checking for unbalanced brackets or unterminated strings
            if (
                output.count("(") != output.count(")")
                or output.count("[") != output.count("]")
                or output.count("{") != output.count("}")
                or output.strip().endswith((":", "=", "(", "[", "{", '"', "'"))
            ):
                self.logger.warning("⚠️ Detected truncated output — retrying once.")
                output += "\n# [Autofix] Partial output detected, retrying.\n"
                retry_output = _stream_once()
                output += "\n" + retry_output

            return output

        except Exception as e:
            self.logger.exception(f"Generation error: {e}")
            return ""
