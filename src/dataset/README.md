# Veritas Dataset

This package contains the core logic to all data related operations for the project.
These operations include:

- Data loading
- Creating main dataset
- Data preprocessing
- Creating train/validation/test splits
- Data loaders for training and evaluation

### How to Use this Package

1. import the package to your package
2. The main function in `src/dataset/process_data.py` will run the entire logic
3. In your package, do a get_dataloader call to get the dataloader for training and validation

**Usage:**
```bash
# Using Poetry script (recommended)
poetry run gather-data

# Or run directly
poetry run python src/dataset/process_data.py
```


