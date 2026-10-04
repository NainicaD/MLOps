# MLOps

Hands-on labs for practising MLOps tools and workflows: automated testing, CI/CD,
and running ML code reliably outside a notebook. Each lab is self-contained, with its
own code, tests, requirements and README.

## Labs

| Lab | What it covers |
| --- | --- |
| [GitHub Actions / Lab 1](Github%20Actions/Lab%201/) | Unit tests with pytest and unittest for a module of similarity and distance measures (cosine, Euclidean, Manhattan, Pearson, Jaccard, Wasserstein), run automatically by GitHub Actions on every push and pull request |

## Repository layout

```
MLOps/
├── .github/workflows/     <- GitHub Actions workflows for all labs
├── Github Actions/
│   └── Lab 1/             <- each lab has its own folder and README
├── .gitignore
└── README.md
```

All workflows live in `.github/workflows/` at the repository root, because GitHub
only runs workflows from there. Each workflow runs inside its own lab folder, and only
starts when that lab's files change.

## Getting started

Create one virtual environment at the repository root and reuse it for every lab:

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
```

Then follow the README inside the lab you want to run. It lists that lab's
dependencies and test commands.
