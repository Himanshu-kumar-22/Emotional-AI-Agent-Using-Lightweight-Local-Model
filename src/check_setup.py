"""
src/check_setup.py
==================
Environment sanity check for the Emotional AI Agent.

Run this script to verify that:
  1. All required Python packages are importable
  2. The correct compute device is detected
  3. Ollama is running and reachable
  4. Required Ollama models are available
  5. The project structure is complete
  6. Configuration is loading correctly

Usage:
    python3 -m src.check_setup
    # or
    python3 src/check_setup.py

Exit codes:
    0 = All checks passed
    1 = One or more checks failed
"""

import sys
import os
from pathlib import Path

# Add project root to path so we can import from config/
# This handles running the script from any directory
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

# ── Formatting helpers ────────────────────────────────────────────────────────
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
BOLD = "\033[1m"
RESET = "\033[0m"


def pass_check(name: str, detail: str = ""):
    detail_str = f" — {detail}" if detail else ""
    print(f"  {GREEN}✓{RESET} {name}{BLUE}{detail_str}{RESET}")


def fail_check(name: str, detail: str = ""):
    detail_str = f" — {detail}" if detail else ""
    print(f"  {RED}✗{RESET} {name}{RED}{detail_str}{RESET}")


def warn_check(name: str, detail: str = ""):
    detail_str = f" — {detail}" if detail else ""
    print(f"  {YELLOW}⚠{RESET} {name}{YELLOW}{detail_str}{RESET}")


def section(title: str):
    print(f"\n{BOLD}{BLUE}{'─' * 50}{RESET}")
    print(f"{BOLD}{title}{RESET}")
    print(f"{BOLD}{BLUE}{'─' * 50}{RESET}")


# ── Check functions ───────────────────────────────────────────────────────────
def check_python_version() -> bool:
    section("Python Version")
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"

    if version.major == 3 and version.minor >= 10:
        pass_check(f"Python {version_str}", "3.10+ required")
        return True
    else:
        fail_check(f"Python {version_str}", "Need Python 3.10 or higher")
        return False


def check_core_packages() -> bool:
    section("Core Package Imports")
    packages = [
        ("torch", "PyTorch — ML framework"),
        ("transformers", "HuggingFace Transformers"),
        ("datasets", "HuggingFace Datasets"),
        ("sklearn", "scikit-learn"),
        ("numpy", "NumPy"),
        ("streamlit", "Streamlit UI"),
        ("cryptography", "Cryptography (AES-256)"),
        ("dotenv", "python-dotenv"),
        ("requests", "HTTP client"),
        ("psutil", "System monitoring"),
        ("yaml", "PyYAML"),
        ("tqdm", "Progress bars"),
        ("sqlite3", "SQLite (stdlib)"),
    ]

    all_passed = True
    for package_name, description in packages:
        try:
            module = __import__(package_name)
            version = getattr(module, "__version__", "unknown")
            pass_check(description, f"v{version}")
        except ImportError as e:
            fail_check(description, f"MISSING — run: pip install {package_name}")
            all_passed = False

    return all_passed


def check_torch_device() -> bool:
    section("Compute Device Detection")

    try:
        import torch

        # Check CUDA
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            pass_check("CUDA GPU", f"{gpu_name} ({vram:.1f}GB VRAM)")
            return True

        # Check MPS (Apple Silicon)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            pass_check("Apple Silicon MPS", "Metal GPU acceleration active")
            # Verify MPS actually works with a small tensor operation
            try:
                x = torch.tensor([1.0, 2.0, 3.0], device="mps")
                _ = x * 2
                pass_check("MPS Tensor Test", "Basic computation verified")
            except Exception as e:
                warn_check("MPS Tensor Test", f"MPS available but test failed: {e}")
            return True

        # CPU fallback
        warn_check(
            "CPU only",
            "No GPU detected — inference will be slower but functional"
        )
        return True

    except ImportError:
        fail_check("PyTorch", "Cannot check device — PyTorch not installed")
        return False


def check_ollama() -> bool:
    section("Ollama Service")
    import requests

    # Check if Ollama server is running
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            pass_check("Ollama server", "Running at http://localhost:11434")
        else:
            warn_check("Ollama server", f"Unexpected status: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        fail_check(
            "Ollama server",
            "Not running — start with: ollama serve"
        )
        return False
    except Exception as e:
        fail_check("Ollama server", str(e))
        return False

    # Check required models
    try:
        data = response.json()
        available_models = [m["name"] for m in data.get("models", [])]

        required_models = ["phi3:mini", "mistral:latest"]
        all_models_present = True

        for model in required_models:
            # Check with and without :latest suffix
            model_base = model.split(":")[0]
            found = any(
                model_base in m or m == model
                for m in available_models
            )
            if found:
                pass_check(f"Model: {model}", "Available")
            else:
                warn_check(
                    f"Model: {model}",
                    f"Not found — run: ollama pull {model}"
                )
                all_models_present = False

        return all_models_present

    except Exception as e:
        warn_check("Model check", f"Could not parse model list: {e}")
        return True  # Server is running, models might just not be listed yet


def check_project_structure() -> bool:
    section("Project Structure")

    required_paths = [
        ("config/settings.py", "Configuration module"),
        ("config/model_config.yaml", "Model configuration"),
        ("src/__init__.py", "Source package"),
        ("src/emotion/__init__.py", "Emotion module"),
        ("src/smoothing/__init__.py", "Smoothing module"),
        ("src/llm/__init__.py", "LLM module"),
        ("src/storage/__init__.py", "Storage module"),
        ("src/pipeline/__init__.py", "Pipeline module"),
        ("app/__init__.py", "App package"),
        (".env", "Environment configuration"),
        (".gitignore", "Git ignore rules"),
        ("requirements.txt", "Dependencies"),
    ]

    all_present = True
    for rel_path, description in required_paths:
        full_path = PROJECT_ROOT / rel_path
        if full_path.exists():
            pass_check(description, rel_path)
        else:
            fail_check(description, f"Missing: {rel_path}")
            all_present = False

    return all_present


def check_configuration() -> bool:
    section("Configuration Loading")

    try:
        from config.settings import settings
        pass_check("Settings loaded", settings.app_name)
        pass_check("Device configured", settings.device)
        pass_check("Emotion model", settings.emotion_model_type)
        pass_check("LLM model", settings.llm_model_name)
        pass_check("Project root", str(settings.project_root))
        pass_check("Data directory", str(settings.data_dir))
        return True
    except Exception as e:
        fail_check("Configuration loading", str(e))
        return False


def check_system_resources() -> bool:
    section("System Resources")

    try:
        import psutil
        import platform

        # Platform info
        system = platform.system()
        machine = platform.machine()
        pass_check("Platform", f"{system} {machine}")

        # RAM
        ram_total = psutil.virtual_memory().total / (1024**3)
        ram_available = psutil.virtual_memory().available / (1024**3)

        if ram_total >= 16:
            pass_check("RAM Total", f"{ram_total:.1f} GB — Excellent (Mistral-7B supported)")
        elif ram_total >= 8:
            pass_check("RAM Total", f"{ram_total:.1f} GB — Good (Mistral-7B with caution)")
        elif ram_total >= 4:
            warn_check("RAM Total", f"{ram_total:.1f} GB — Limited (Phi-3-Mini only)")
        else:
            fail_check("RAM Total", f"{ram_total:.1f} GB — Insufficient")

        pass_check("RAM Available", f"{ram_available:.1f} GB currently free")

        # CPU
        cpu_count = psutil.cpu_count(logical=False)
        cpu_freq = psutil.cpu_freq()
        freq_str = f"{cpu_freq.current:.0f}MHz" if cpu_freq else "unknown"
        pass_check("CPU Cores", f"{cpu_count} physical cores at {freq_str}")

        return True

    except Exception as e:
        fail_check("System resources", str(e))
        return False


# ── Main runner ───────────────────────────────────────────────────────────────
def main() -> int:
    print(f"\n{BOLD}{'=' * 50}{RESET}")
    print(f"{BOLD}  Emotional AI Agent — Environment Check{RESET}")
    print(f"{BOLD}{'=' * 50}{RESET}")

    results = {
        "Python Version": check_python_version(),
        "Core Packages": check_core_packages(),
        "Compute Device": check_torch_device(),
        "Ollama Service": check_ollama(),
        "Project Structure": check_project_structure(),
        "Configuration": check_configuration(),
        "System Resources": check_system_resources(),
    }

    # Summary
    section("Summary")
    all_passed = True
    for check_name, passed in results.items():
        if passed:
            pass_check(check_name)
        else:
            fail_check(check_name)
            all_passed = False

    print()
    if all_passed:
        print(f"{GREEN}{BOLD}✓ All checks passed. Ready to begin Phase 1.{RESET}\n")
        return 0
    else:
        failed = [name for name, passed in results.items() if not passed]
        print(f"{RED}{BOLD}✗ {len(failed)} check(s) failed: {', '.join(failed)}{RESET}")
        print(f"{YELLOW}  Resolve the issues above before proceeding.{RESET}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())