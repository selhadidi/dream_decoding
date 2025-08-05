

# EEG Dream Decoding

## Overview

Understanding and decoding the content of dreams from brain activity remains one of the most intriguing and underexplored frontiers in neuroscience and artificial intelligence. This project investigates the decoding of high-level dream content categories using non-invasive EEG recordings collected during sleep. Building on the Zhang and Wamsley (2019) dataset, we manually annotated each dream report with a single label from a set of 13 high-level semantic content classes.

Our experimental pipeline evaluates two key tasks:
1. **EEG-based content classification**
2. **Report retrieval**

We demonstrate that both tasks can be learned above chance, indicating that EEG signals contain meaningful information about dream content. This work introduces a novel framework for decoding subjective experience and represents a step toward interpretable, content-aware dream analysis using machine learning.

---

## Repository Structure

- `notebooks/01_classify_eeg_to_topic.ipynb` — RQ1: EEG-based content classification
- `notebooks/02_retreive_eeg_to_report.ipynb` — RQ2: Report retrieval
- `data/` — Preprocessed features and reports (CSV files)
- `src/utils/label_maps.py` — Mapping of trial IDs to semantic labels
- `environment/environment.yml` — Conda environment for reproducibility

---

## Quickstart

1. **Clone the repository and set up the environment:**
   ```bash
   conda env create -f environment/environment.yml
   conda activate eeg-py39
   ```

2. **Run the notebooks:**
   - Open the desired notebook in `notebooks/` (e.g., `01_classify_eeg_to_topic.ipynb` for RQ1).
   - The **first cell** in each notebook performs training and validation (customizable input).
   - The **second cell** tests on the hold-out set.

3. **Data:**
   - All necessary preprocessed features and reports are provided in the `data/` directory.
   - Dream theme (label) information is in `data/merged_embedded_reports.csv`.
   - Mapping of trial IDs to labels is in `src/utils/label_maps.py`.

---

## Data

- The repository does **not** include the full raw EEG dataset, but provides all extracted features and reports (+ embeddings)required for the tasks.
- The CSV files in `data/` are ready to use, no extra preprocessing is required.

---

## Expected Results

- **RQ1 (Classification):**
  - Metrics: Top-1 accuracy, F1 score, confusion matrices
  - Results are reported per fold and as the mean of three folds

- **RQ2 (Retrieval):**
  - Metrics: Top-1, Top-3 accuracy, Mean Squared Error (MSE)
  - Results are reported per fold and as the mean of three folds

- Example outputs and demo results can be found in `notebooks/demo_outputs_rq2/`.

---

## Customization

- You can customize the input and parameters in the first cell of each notebook to experiment with different settings.

---

## Acknowledgements

- This project builds on the Zhang and Wamsley (2019) dataset.

