# GitHub Actions

Labs for practising GitHub Actions, GitHub's built-in automation platform. A workflow
is a YAML file that tells GitHub what to run (for example, a test suite) and when to
run it (for example, on every push or pull request). GitHub then runs it on its own
servers and reports a pass or fail for each commit.

## Key concepts

- **Workflow:** a YAML file in `.github/workflows/` that defines an automated process.
- **Event (trigger):** what starts a workflow, such as a `push` or `pull_request` to `main`.
- **Job:** a set of steps that runs on a fresh virtual machine (here, `ubuntu-latest`).
- **Step:** a single command (`run:`) or a reusable action (`uses:`), such as
  `actions/checkout` or `actions/setup-python`.
- **Matrix:** runs the same job several times with different settings, such as
  several Python versions.
- **Artifact:** a file a workflow saves for download, such as a test report.

## Labs

| Lab | What it covers |
| --- | --- |
| [Lab 1](Lab%201/) | Unit tests with pytest and unittest for a module of similarity and distance measures, run automatically on every push and pull request, with a Python version matrix and test-report artifacts |

## How the workflows are organized

GitHub only runs workflows stored in `.github/workflows/` at the repository root, so
every lab's workflows live there rather than inside the lab folder. Each workflow:

- is named after its lab (e.g. `lab1_pytest.yml`, `lab1_unittest.yml`)
- runs its commands inside its own lab folder, using `working-directory`
- only starts when files in that lab folder (or the workflow file itself) change,
  using a `paths` filter

This keeps labs independent: changing one lab doesn't re-run another lab's tests.

## Viewing results

Open the repository's **Actions** tab. Each workflow run shows its jobs with a green
check (passed) or a red cross (failed). Click a job, then expand a step, to see its
full output.
