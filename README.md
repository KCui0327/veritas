# veritas
APS360 Project - Fake News Detection System

## Overview

Veritas is a comprehensive fake news detection system that uses multiple machine learning approaches to classify news articles as real or fake. The project includes:

- **Base Model**: Traditional machine learning approach using TF-IDF and Logistic Regression
- **RNN Model**: Deep learning approach using Recurrent Neural Networks
- **Dataset Processing**: Automated pipeline to process multiple fake news datasets

## Poetry Scripts

The project provides convenient Poetry scripts for common tasks:

| Script | Command | Description |
|--------|---------|-------------|
| `gather-data` | `poetry run gather-data` | Process and combine multiple fake news datasets into a unified dataset |
| `train-rnn` | `poetry run train-rnn` | Train the RNN model for fake news detection |
| `train-base` | `poetry run train-base` | Train the base model using TF-IDF and Logistic Regression |
| `visualize-rnn` | `poetry run visualize-rnn` | Generate visualizations of RNN training results |
| `visualize-base` | `poetry run visualize-base` | Generate visualizations of base model performance |

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

### Project Dependencies

The project uses the following main dependencies (defined in `pyproject.toml`):
- **torch**: PyTorch for deep learning models
- **pandas**: Data manipulation and analysis
- **scikit-learn**: Machine learning algorithms
- **matplotlib**: Plotting and visualization
- **seaborn**: Statistical data visualization

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

## Running the Project

The project contains three main components, each with its own main.py file:

### 1. Dataset Processing (`src/dataset/process_data.py`)

This script processes multiple fake news datasets and combines them into a unified Veritas dataset.

**Usage:**
```bash
poetry run python src/dataset/process_data.py
# Or use the Poetry script:
poetry run gather-data
```

**What it does:**
- Runs all processor scripts in the `processor/` directory
- Combines outputs from multiple datasets (FakeNewsNet, ISOT, LIAR, Politifact, WELFake)
- Removes duplicate entries based on content hashing
- Creates a unified `veritas_dataset.csv` file
- Initializes a `VeritasDataset` instance for PyTorch training

**Output:**
- `veritas_dataset.csv`: Combined dataset with columns: `statement`, `verdict`, `id`

### 2. Base Model (`src/base_model/main.py`)

This script trains and tests a traditional machine learning model using TF-IDF features and Logistic Regression.

**Usage:**
```bash
poetry run python src/base_model/main.py
```

**What it does:**
- Trains a Logistic Regression model on TF-IDF features
- Performs model inference on test articles
- Creates comprehensive visualizations of model performance
- Saves the trained model for later use

**Features:**
- TF-IDF vectorization with n-gram features
- Cross-entropy loss for binary classification
- Model persistence and loading
- Real-time inference on new articles
- Performance visualizations (confusion matrix, metrics, etc.)

### 3. RNN Model (`src/rnn_model/main.py`)

This script trains a deep learning model using Recurrent Neural Networks for sequence-based fake news detection.

**Usage:**
```bash
poetry run python src/rnn_model/main.py
```

**What it does:**
- Initializes an RNN model for text classification
- Configures training parameters (optimizer, loss function, etc.)
- Trains the model using the custom trainer
- Saves checkpoints and training history

**Features:**
- Custom RNN architecture for text processing
- Configurable training parameters via `TrainingConfig`
- Automatic checkpoint saving
- Training history tracking
- GPU support when available

**Configuration:**
The training can be customized by modifying the `TrainingConfig` in `main.py`:
- Learning rate: 0.001
- Epochs: 100
- Batch size: 32
- Optimizer: Adam
- Loss function: CrossEntropyLoss
- Save directory: "checkpoints"

### Using Poetry Scripts (Recommended)

The project provides convenient Poetry scripts for common tasks. These are the recommended way to run the project:

1. **Process and gather data:**
   ```bash
   poetry run gather-data
   ```

2. **Train the RNN model:**
   ```bash
   poetry run train
   ```

3. **Visualize training results:**
   ```bash
   poetry run visualize
   ```

**Note:** The base model doesn't have a Poetry script yet, so you'll need to run it directly:
```bash
poetry run python src/base_model/main.py
```

### Alternative: Direct Python Execution

You can also run the Python files directly if you prefer:

1. **Process the dataset:**
   ```bash
   poetry run python src/dataset/process_data.py
   ```

2. **Train the base model:**
   ```bash
   poetry run python src/base_model/main.py
   ```

3. **Train the RNN model:**
   ```bash
   poetry run python src/rnn_model/main.py
   ```

4. **Visualize RNN training results:**
   ```bash
   poetry run python src/rnn_model/visualize.py
   ```

### Running All Components

To run the complete pipeline using Poetry scripts:

1. **Process the dataset first:**
   ```bash
   poetry run gather-data
   ```

2. **Train the base model:**
   ```bash
   poetry run python src/base_model/main.py
   ```

3. **Train the RNN model:**
   ```bash
   poetry run train
   ```

4. **Visualize the RNN training results:**
   ```bash
   poetry run visualize
   ```

### Alternative: Using Poetry Shell

You can also activate the Poetry environment once and run commands directly:

```bash
# Activate the Poetry environment
poetry shell

# Now you can run Poetry scripts without 'poetry run'
gather-data
train
visualize

# Or run Python files directly
python src/base_model/main.py
python src/dataset/process_data.py
python src/rnn_model/main.py

# Deactivate when done
exit
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
