Project development guidelines (Assistantship)

Audience: This document targets contributors familiar with Python ML workflows, Jupyter, and basic packaging/testing. It focuses on repo‑specific caveats and reliable ways to run and test code in this project.

1) Build and configuration

Python version
- Recommended: Python 3.11 or 3.12. The pinned stack (scikit_learn==1.7.2, xgboost==3.0.5, pandas==2.3.x, matplotlib==3.10.x, seaborn==0.13.x) works on 3.11+.

Create an isolated environment (Windows PowerShell)
- py -3.11 -m venv .venv
- .\.venv\Scripts\Activate.ps1
- python -m pip install --upgrade pip
- pip install -r requirements.txt

Jupyter usage
- Notebooks in this repo (wine_dataset.ipynb, concrete_dataset.ipynb, WIP_superconductivity_dataset.ipynb) assume an interactive kernel with display() available (IPython). If running headless/CI, install Jupyter:
  - pip install jupyterlab ipykernel
  - python -m ipykernel install --user --name assistantship
- Matplotlib in headless contexts may need a non‑GUI backend:
  - import matplotlib; matplotlib.use("Agg")  # before pyplot imports, or set MPLBACKEND=Agg

Data access and network
- Both concrete_dataset.py and wine_dataset.py fetch UCI datasets via ucimlrepo at import time. This requires internet access. If you are behind a proxy, configure HTTPS_PROXY.
- To avoid repeated network calls during exploratory runs, consider caching the fetched dataset objects to disk (e.g., using joblib dump/load) in a future refactor.

Reproducibility
- random_state is set to 1 in both scripts. Maintain or thread this through new code paths to keep results comparable.

2) Testing: how to configure and run

Important: Avoid importing the top‑level scripts in tests
- The .py files contain heavy, side‑effectful code at module import (dataset fetch, model fitting, plotting). Importing them will trigger long‑running jobs and require a display backend. Until the code is refactored (see below), tests should either:
  - Inspect files as text instead of importing, or
  - Import only small refactored utilities placed in import‑safe modules.

Using the standard library (unittest)
- No extra dependencies are required. Place tests under tests/ using the pattern test_*.py.
- Run all tests:
  - python -m unittest discover -s tests -p "test*.py" -v

Example: simple smoke test used to validate test execution locally
- File: tests/test_repository_smoke.py
  - Checks for presence of key function definitions in source files (no imports).
  - Verifies requirements contain expected packages.
- Content example:
  - import os, unittest
  - REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
  - class TestRepositorySmoke(unittest.TestCase):
      - def read_file(self, p):
        - with open(os.path.join(REPO_ROOT, p), "r", encoding="utf-8") as f: return f.read()
      - def test_concrete_has_functions(self):
        - src = self.read_file("concrete_dataset.py")
        - for token in ["def objective_function(", "def objective_function_xgb(", "def simulate_human_expert(", "def acquisition_function("]:
          - self.assertIn(token, src)
      - def test_requirements_core(self):
        - req = self.read_file("requirements.txt")
        - for pkg in ["ucimlrepo", "scikit-optimize", "pandas", "scikit_learn", "xgboost"]:
          - self.assertIn(pkg, req)
- Execution (local verification):
  - python -m unittest discover -s tests -p "test*.py" -v
  - Outcome observed: Ran 3 tests, OK.

Optional: pytest
- If you prefer pytest, add it to your dev environment (not pinned in requirements.txt):
  - pip install pytest
- Run: pytest -q
- Keep the same rule: do not import the heavy top‑level scripts until they are made import‑safe.

Guidelines for adding new tests
- Prefer pure functions and small utilities in separate modules that are safe to import, e.g., utils/data_loading.py, utils/metrics.py. Export these from __init__.py for clear test targets.
- Structure
  - project_root/
    - utils/
      - data_loading.py   # import‑safe helpers (no plotting or network on import)
    - tests/
      - test_data_loading.py
- Keep tests deterministic: set numpy/random/random_state seeds; avoid network by mocking (e.g., unittest.mock.patch) or by passing precomputed fixtures.

3) Additional development information

Refactor recommendation: make scripts import‑safe
- Both concrete_dataset.py and wine_dataset.py execute data fetching, plotting, and model training at the top level. This limits reusability and testability. A minimal, low‑risk pattern for future edits:
  - Move heavy logic into functions (e.g., load_data(), fit_models(X, y), make_plots(...)).
  - Guard execution under if __name__ == "__main__": to allow import without side effects.
  - Example:
    - def main():
      - X, y = load_data()
      - model = fit_models(X, y)
      - make_plots(...)
    - if __name__ == "__main__":
      - main()

Jupyter‑specific calls
- display(...) is used in scripts. This is an IPython utility; it is undefined when running as a plain Python script. Either:
  - Gate with a runtime check (from IPython.display import display inside a try/except), or
  - Move display calls into notebook‑only code paths.

Hyperparameter optimization notes
- BayesSearchCV (scikit‑optimize) uses scikit‑learn estimators and can be CPU‑intensive with n_jobs = -1. In shared environments, set n_jobs to a bounded value or expose it as a parameter.
- scikit‑optimize is unpinned. If compatibility issues arise with recent scikit‑learn, try pinning to scikit-optimize==0.9.0.

Plotting
- For non‑interactive runs, avoid plt.show() in library code. Return figures or save to files (e.g., figs/*.png) and let the caller decide what to do.

Data integrity
- UCI datasets occasionally change or get mirrored. Consider snapshotting a copy of the raw CSVs in data/raw/ and writing a small loader that prefers local cache when available.

Coding style
- Follow PEP 8; black/ruff are recommended for consistency:
  - pip install black ruff
  - black .
  - ruff check --fix .

Known gotchas in this repo
- Importing wine_dataset.py or concrete_dataset.py will trigger network/downloads and long‑running model fits; avoid module import in tests.
- display(...) requires IPython; not available in pure Python execution.
- Matplotlib backends can break in headless CI unless MPLBACKEND is set to Agg.

Appendix: Clean-up policy for this doc’s examples
- Any temporary files created to validate the testing instructions (e.g., tests/test_repository_smoke.py) should not be committed permanently. They are safe to recreate on demand following the steps above.
