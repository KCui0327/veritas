# veritas

APS360 Project - Fake News Detection System

## Overview

Veritas is a comprehensive fake news detection system that uses multiple machine learning approaches to classify news articles as real or fake. The project includes:

- **Base Model**: Traditional machine learning approach using TF-IDF and Logistic Regression
- **RNN Model**: Deep learning approach using Recurrent Neural Networks
- **Dataset Processing**: Automated pipeline to process multiple fake news datasets

## Poetry Scripts

The project provides convenient Poetry scripts for common tasks:

| Script           | Command                     | Description                                                            |
| ---------------- | --------------------------- | ---------------------------------------------------------------------- |
| `gather-data`    | `poetry run gather-data`    | Process and combine multiple fake news datasets into a unified dataset |
| `train-rnn`      | `poetry run train-rnn`      | Train the RNN model for fake news detection                            |
| `train-base`     | `poetry run train-base`     | Train the base model using TF-IDF and Logistic Regression              |
| `visualize-rnn`  | `poetry run visualize-rnn`  | Generate visualizations of RNN training results                        |
| `visualize-base` | `poetry run visualize-base` | Generate visualizations of base model performance                      |

These scripts are defined in `pyproject.toml` and provide a simple interface to the project's main functionality.

## Getting Started

This guide will help you set up the development environment for the veritas project.

### Prerequisites

- Python 3.10 or higher
- Poetry (Python dependency management tool)

### Installing Poetry

If you don't have Poetry installed, follow the official installation guide:

**On macOS/Linux:**

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

**On Windows:**

```powershell
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -
```

**Verify installation:**

```bash
poetry --version
```

### Setting up the Project

1. **Clone the repository** (if you haven't already):

   ```bash
   git clone <repository-url>
   cd veritas
   ```

2. **Install dependencies using Poetry:**

   ```bash
   poetry install
   ```

3. **Activate the Poetry virtual environment:**

   ```bash
   poetry shell
   ```

4. **Verify the environment is active:**
   ```bash
   which python  # Should point to Poetry's virtual environment
   pip list      # Should show all project dependencies
   ```

### Code Linting with Black

This project uses [Black](https://black.readthedocs.io/) for automatic code formatting to maintain consistent code style.

#### Installing Black

1. **Install Black** (if not already installed):

   ```bash
   poetry add --group dev black
   ```

2. **Verify installation:**
   ```bash
   poetry run black --version
   ```

#### Using Black with VSCode Extension

The easiest way to use Black is with the VSCode extension:

1. **Install the Black Formatter extension** in VSCode:

   - Open VSCode
   - Go to Extensions (Ctrl+Shift+X or Cmd+Shift+X)
   - Search for "Black Formatter"
   - Install the extension by Microsoft

2. **Configure VSCode to use Black**:

   - Open VSCode settings (Ctrl+, or Cmd+,)
   - Search for "python formatting provider"
   - Set it to "black"
   - Enable "Format on Save" for automatic formatting

3. **Format your code**:
   - Right-click in a Python file and select "Format Document"
   - Or use the keyboard shortcut: Shift+Alt+F (Windows/Linux) or Shift+Option+F (Mac)
   - Or enable "Format on Save" to automatically format when you save files

#### Command Line Usage (Optional)

If you prefer using Black from the command line:

```bash
# Format all Python files in the project
poetry run black .

# Format a specific file
poetry run black src/base_model/main.py

# Check what would be formatted without making changes
poetry run black --check .
```

### Deactivating the Poetry Environment

When you're done working on the project:

```bash
exit  # If using poetry shell
# or
deactivate  # If the environment is still active
```

### Troubleshooting

- **If you get permission errors**: Make sure you have write permissions in the project directory
- **If Poetry fails to install dependencies**: Try updating Poetry first: `poetry self update`
- **If you need to recreate the environment**: Delete the `.venv` folder and run `poetry install` again
- **If Black formatting fails**: Make sure you're in the Poetry environment and Black is installed
- **If packages are missing**: Run `poetry install` to ensure all dependencies are installed

### Notes

- Always use `poetry run` or activate the Poetry environment before running scripts
- Poetry automatically manages virtual environments and dependencies
- The `pyproject.toml` file contains all project dependencies and configuration
- Run `poetry run black .` before committing to ensure consistent code formatting
- Poetry creates a `.venv` directory in your project folder for the virtual environment
