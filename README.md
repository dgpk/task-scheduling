# Evol-task-sched

## Installation
*   Setup a virtual environment (`venv`) with Python 3.7+
*   Run `python -m pip install -r requirements.txt` to install dependencies.

## Development Setup (Code Formatting)

This project uses `black` for code formatting and `pre-commit` to enforce it automatically before each commit.

To set it up, run this one-time command after installation:

```bash
pre-commit install
```

From now on, `black` will automatically format your code every time you make a commit. If it modifies a file, you will need to `git add` the changes and commit again.
