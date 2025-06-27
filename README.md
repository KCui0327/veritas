# veritas
APS360 Project

## Getting Started

This guide will help you set up the development environment for the veritas project.

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Setting up the Virtual Environment

1. **Clone the repository** (if you haven't already):
   ```bash
   git clone <repository-url>
   cd veritas
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**:

   **On macOS/Linux**:
   ```bash
   source venv/bin/activate
   ```

   **On Windows**:
   ```bash
   venv\Scripts\activate
   ```

4. **Verify the virtual environment is active**:
   ```bash
   which python  # Should point to your venv directory
   pip list      # Should show only basic packages
   ```

### Installing Dependencies

1. **Install required packages** (if you have a requirements.txt file):
   ```bash
   pip install -r requirements.txt
   ```

2. **Or install packages individually** (if no requirements.txt exists):
   ```bash
   pip install <package-name>
   ```

3. **Verify installation**:
   ```bash
   pip list
   ```

### Code Linting with Black

This project uses [Black](https://black.readthedocs.io/) for automatic code formatting to maintain consistent code style.

#### Installing Black

1. **Install Black** (if not already installed):
   ```bash
   pip install black
   ```

2. **Verify installation**:
   ```bash
   black --version
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
black .

# Format a specific file
black base_model/base_model.py

# Check what would be formatted without making changes
black --check .
```

### Running the Project

1. **Make sure your virtual environment is activated**:
   ```bash
   source venv/bin/activate  # macOS/Linux
   # or
   venv\Scripts\activate     # Windows
   ```

2. **Run your project**:
   ```bash
   python base_model/base_model.py
   ```

### Deactivating the Virtual Environment

When you're done working on the project:
```bash
deactivate
```

### Troubleshooting

- **If you get permission errors**: Make sure you have write permissions in the project directory
- **If packages fail to install**: Try upgrading pip first: `pip install --upgrade pip`
- **If you need to recreate the environment**: Delete the `venv` folder and repeat the setup steps
- **If Black formatting fails**: Make sure you're in the virtual environment and Black is installed

### Notes

- Always activate the virtual environment before working on the project
- The virtual environment keeps project dependencies isolated from your system Python
- Remember to add `venv/` to your `.gitignore` file if it's not already there
- Run `black .` before committing to ensure consistent code formatting
