from __future__ import annotations

import importlib
import platform
import sys

PACKAGES = [
    "numpy",
    "pandas",
    "sklearn",
    "lightgbm",
    "catboost",
    "xgboost",
    "optuna",
    "shap",
    "matplotlib",
]


def main() -> None:
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    failed = []
    for package in PACKAGES:
        try:
            module = importlib.import_module(package)
            version = getattr(module, "__version__", "bilinmiyor")
            print(f"OK  {package}: {version}")
        except Exception as exc:  # environment diagnostics should report every failure
            failed.append((package, str(exc)))
            print(f"FAIL {package}: {exc}")
    if failed:
        raise SystemExit("Eksik veya hatalı paketler var.")
    print("Ortam hazır.")


if __name__ == "__main__":
    main()
