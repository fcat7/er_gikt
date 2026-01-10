# 重构结构（2026-01-06 23:09） - Refactoring Guide

We have refactored the codebase to support multiple datasets (e.g., `assist09`, `assist12`) with isolated data processing directories.

## Changes Summary

### 1. Directory Structure Update
The system now expects data to be organized in subdirectories under `kt/data/`:
```
kt/data/
  assist09/      # Generated via: DATASET=assist09 python data_process.py
    user_seq.npy
    qs_table.npz
    ...
  assist12/      # Generated via: DATASET=assist12 python data_process.py
    ...
```

### 2. Code Changes

*   **`kt/config.py`**: 
    *   Renamed `PROCESSED_DATA_DIR` (class attr) to `PROCESSED_DATA_ROOT`.
    *   Added **`PROCESSED_DATA_DIR`** (property) that dynamically returns `kt/data/{DATASET_NAME}` based on the active configuration.

*   **`kt/dataset.py`**:
    *   `UserDataset` now requires a `config` object during initialization.
    *   Loads files relative to `config.PROCESSED_DATA_DIR`.

*   **`kt/train_test.py`**:
    *   Reads `DATASET` environment variable (default: `assist09`).
    *   Loads the specific dataset configuration via `get_config()`.
    *   Passes the correct paths to `build_adj_list` and `UserDataset`.

*   **`kt/utils.py`**:
    *   `build_adj_list` now accepts a `data_dir` argument instead of hardcoding `data/`.

*   **`kt/data_process.py`**:
    *   Reads `DATASET` environment variable to determine which dataset to process.
    *   Automatically creates the subdirectory (e.g., `kt/data/assist12`) if it doesn't exist.

## How to Run

### Step 1: Data Preprocessing (Required for new structure)

You **must** re-run data processing to generate files in the new directory structure (or manually move your existing files from `kt/data/` to `kt/data/assist09/`).

```bash
# Process Assist09 (Default)
DATASET=assist09 python kt/data_process.py

# Process Assist12 (If configured in config.py)
DATASET=assist12 python kt/data_process.py
```

### Step 2: Training

```bash
# Train on Assist09
DATASET=assist09 python kt/train_test.py

# Train on Assist12
DATASET=assist12 python kt/train_test.py
```

## Migration Note

If you have existing processed data in `kt/data/*.npy` that you want to keep without re-processing, please move them:

```bash
mkdir -p kt/data/assist09
mv kt/data/*.npy kt/data/assist09/
mv kt/data/*.npz kt/data/assist09/
```
