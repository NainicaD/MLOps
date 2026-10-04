# Lab 1: Similarity Measures with CI

This lab is used to practice setting up GitHub Actions: automated tests that run on
GitHub's servers every time code is pushed or a pull request is opened. The code being
tested is a Python module of similarity and distance measures, with tests written in
both pytest and unittest.

## What's in the lab

`src/similarity.py` implements six measures in plain Python:

- **Cosine similarity:** angle between two vectors, ignoring their length; the standard
  way to compare embeddings. Range [-1, 1].
- **Euclidean / Manhattan distance:** straight-line (L2) and sum-of-differences (L1)
  distance; used in clustering and nearest-neighbour search.
- **Pearson correlation:** linear relationship between two variables. Range [-1, 1].
- **Jaccard similarity:** overlap between two sets (shared items / all items). Range [0, 1].
- **Wasserstein distance:** how much "work" it takes to turn one sample's distribution
  into another's. Used to detect data drift between training data and live data.

Invalid input raises an error instead of returning a misleading number: vectors of
different lengths, empty or non-numeric input, a zero vector for cosine, a constant
vector for Pearson, two empty sets for Jaccard, and negative or all-zero weights for
Wasserstein.

`data/sample_data.json` holds the data the tests use: word-count vectors for three
short documents, and feature samples for a data-drift check.

The tests check:

- values worked out by hand and reference values for Wasserstein distance
- properties every measure must satisfy: symmetry, identical inputs giving distance 0
  or similarity 1, scale invariance of cosine similarity, the triangle inequality for
  Euclidean distance, and the shift property of Wasserstein distance
- that invalid input raises the right errors

## Folder structure

The lab lives at `Github Actions/Lab 1/` inside the `MLOps` repository. Its workflows
are at the repository root in `.github/workflows/`, because GitHub only runs workflows
from there.

```
MLOps/
├── .github/workflows/
│   ├── lab1_pytest.yml
│   └── lab1_unittest.yml
└── Github Actions/
    └── Lab 1/
        ├── data/
        │   ├── __init__.py
        │   └── sample_data.json
        ├── src/
        │   ├── __init__.py
        │   └── similarity.py
        ├── test/
        │   ├── __init__.py
        │   ├── test_pytest.py
        │   └── test_unittest.py
        ├── README.md
        ├── pytest.ini
        └── requirements.txt
```

## Running the tests

### 1. Set up the environment (once)

From the repository root (`MLOps/`):

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
cd "Github Actions/Lab 1"
pip install -r requirements.txt
```

Check that the virtual environment is active:

```bash
which python                       # should end in MLOps/.venv/bin/python
```

If it points to conda or miniconda instead, run `conda deactivate` and activate
`.venv` again.

### 2. Run the tests

Run these from inside `Github Actions/Lab 1` (the folder names contain spaces,
so keep the quotes when you `cd` into it):

```bash
pytest -v                                  # expected: 63 passed
python -m unittest test.test_unittest -v   # expected: Ran 9 tests ... OK
```

`pytest -v` runs both test files: 54 pytest cases plus the 9 unittest tests.
`pytest.ini` sets `pythonpath = .` so that `from src import similarity` resolves
correctly.

### 3. Tests in GitHub Actions

Every push or pull request to `main` that changes this folder runs two workflows:

- **Lab 1 - Testing with Pytest** runs `pytest` on Python 3.10, 3.11 and 3.12 and
  uploads a JUnit XML report for each version (see the run's Artifacts).
- **Lab 1 - Python Unittests** runs the unittest suite on Python 3.11.

Results are in the repository's **Actions** tab.

### Troubleshooting

| Problem | Fix |
| --- | --- |
| `ModuleNotFoundError: No module named 'src'` | Run from inside `Github Actions/Lab 1`, and check that `src/` and `test/` each contain `__init__.py` |
| `pytest` collects fewer than 63 tests | Check that `test/test_pytest.py` isn't empty (`wc -l test/test_pytest.py` should show 240) |
| `cd: string not in pwd` or `too many arguments` | Put the path in quotes: `cd "Github Actions/Lab 1"` |
