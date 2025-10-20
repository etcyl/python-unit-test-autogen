# Auto Test Generator

This project provides a local and free system for generating Python unit tests using a local large language model (LLM) such as Llama 3 or Mistral. It automatically analyzes your repository, generates pytest tests, runs them, and iteratively fixes failures up to a configurable number of times. It also integrates pytest coverage to focus only on untested or under-tested code.

The system runs fully offline and uses llama-cpp-python for CPU inference, making it suitable for local environments on Windows 11 without the need for GPU acceleration or cloud access.

## 1. Prerequisites

Ensure that the following are installed on your system:

1. **Anaconda or Miniconda**
   - Download and install from https://www.anaconda.com/download or https://docs.conda.io/en/latest/miniconda.html
   - Verify installation:
     ```bash
     conda --version
     ```

2. **Git**
   - Download from https://git-scm.com/downloads
   - Verify installation:
     ```bash
     git --version
     ```

3. **Visual Studio Code**
   - Install from https://code.visualstudio.com
   - Install the following extensions inside VS Code:
     - Python
     - Git
     - YAML
     - Markdown All in One
   - Open a Git Bash terminal inside VS Code:
     - Press `Ctrl + Shift + P` → select "Select Default Profile" → choose "Git Bash"
     - Open a new terminal: `Ctrl + Shift + ~`

4. **CMake and Build Tools for llama-cpp**
   - Install using:
     ```bash
     conda install -c conda-forge cmake make
     ```
   - Ensure you have Visual Studio Build Tools (https://visualstudio.microsoft.com/visual-cpp-build-tools/)

## 2. Clone the Repository

In Git Bash inside Visual Studio Code, run:
```bash
git clone https://github.com/your-username/auto-testgen.git
cd auto-testgen
```

## 3. Create and Activate a Conda Environment

Create a clean isolated environment for this project:

```bash
conda create -n autotestgen python=3.10 -y
conda activate autotestgen
```

## 4. Install Project Dependencies

Install all required Python dependencies listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

If you see build errors for `llama-cpp-python`, try:
```bash
pip install --upgrade pip setuptools wheel
pip install llama-cpp-python --prefer-binary
```

## 5. Verify the Installation

Check that all modules were installed correctly:
```bash
python -m pytest --version
python -m llama_cpp
```

If no errors appear, proceed to configuration.

## 6. Configure the Application

The application uses a configuration file named `config.yaml` in the project root. Edit this file before running.

Example configuration:

```yaml
model_path: "models/llama-3-8b-instruct.Q4_K_M.gguf"
repo_path: "C:/Users/YourName/Projects/target_repo"
output_dir: "tests/auto_generated"
max_fix_attempts: 10
temperature: 0.2
max_tokens: 1500
context_window: 8192
log_level: "INFO"
coverage_threshold: 80
metadata_file: "auto_testgen_meta.json"
```

**model_path:** path to the downloaded Llama or Mistral GGUF model file  
**repo_path:** absolute path to the Python repository you want to generate tests for  
**output_dir:** directory for saving generated test files  
**coverage_threshold:** coverage percentage threshold before skipping files  

You can download a compatible quantized model from Hugging Face, such as:
- https://huggingface.co/TheBloke/Llama-3-8B-Instruct-GGUF

Place the model file inside a `models/` directory in the project root.

## 7. Run the Application

To start generating unit tests, run:

```bash
python auto_testgen.py
```

All logs will be saved in `logs/auto_testgen.log`.

The program will:
1. Run pytest with coverage.
2. Identify untested files or functions.
3. Generate new pytest unit tests using the local model.
4. Run the tests and fix failures iteratively up to the configured limit.
5. Save generated test files under `tests/auto_generated/`.
6. Display a summary report with coverage improvements.

## 8. Optional: Run Coverage Manually

You can manually check the coverage report at any time:

```bash
pytest --cov /path/to/your/repo --cov-report=term-missing
```

## 9. Deactivating and Removing the Environment

To deactivate the environment:
```bash
conda deactivate
```

To remove it completely:
```bash
conda env remove -n autotestgen
```

## 10. Troubleshooting

**llama-cpp build error on Windows**
- Ensure you installed Visual Studio Build Tools
- Add CMake and `cl.exe` to PATH
- Reinstall with:
  ```bash
  pip install llama-cpp-python --force-reinstall --prefer-binary
  ```

**Import errors in Visual Studio Code**
- Open Command Palette → "Python: Select Interpreter"
- Choose the `autotestgen` environment

**Memory issues**
- Use a smaller model such as `Llama-3-8B-Instruct.Q4_K_M.gguf` or `Phi-3-mini-4k-instruct.gguf`
- Reduce `context_window` to 4096 in `config.yaml`

## 11. Uninstallation

To remove all installed files, simply delete the project folder and the conda environment:

```bash
conda env remove -n autotestgen
rm -rf auto-testgen
```

## 12. Summary

This project provides an offline solution for automatic Python unit test generation using a local language model. It is designed to integrate easily with existing codebases, improve testing coverage, and accelerate quality assurance workflows.

**End of README**

