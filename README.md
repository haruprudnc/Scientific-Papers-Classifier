# Scientific Papers Classifier

## Overview
The Scientific Papers Classifier is a machine learning project designed to process, analyze, and classify scientific research papers. It utilizes external APIs (DOI and OpenAlex) to enrich paper metadata, processes the raw data into a clean format, and applies a machine learning model to categorize the documents.

## Project Structure
The repository is organized into the following directories and files:

* **`core/`**: Contains scripts for external API integrations.
    * `doi_org.py`: Handles interactions with the DOI.org API.
    * `openalex_org.py`: Handles data fetching and interaction with the OpenAlex API.
* **`models/`**: Stores serialized machine learning models.
    * `model.pkl`: The trained classifier model ready for inference.
* **`modified_csv/`**: Contains the intermediate and final processed datasets.
    * `train_raw_data.csv` / `test_raw_data.csv`: Intermediate raw data extracted from source.
    * `train_preprocess_data.csv` / `test_preprocess_data.csv`: Cleaned and formatted data ready for model training/testing.
* **`src/`**: Contains the original, raw source data files.
    * `Stage_1_publcitrain.csv`: Initial training dataset.
    * `test_(2).csv`: Initial testing dataset.
* **`utils/`**: Utility scripts for terminal output and user experience.
    * `cli_progress_bar.py`: Custom command-line progress bar.
    * `color.py`: Terminal text coloring utility.
    * `terminal.py`: General terminal helper functions.
* **`data_preprocessing.py`**: The main execution script for cleaning the source data and preparing it for the machine learning pipeline.

## Installation

1.  Clone the repository to your local machine:
    ```bash
    git clone https://github.com/haruprudnc/Scientific-Papers-Classifier
    cd scientific-papers-classifier
    ```

2.  Ensure you have Python 3.x installed.

3.  Install the necessary dependencies. (Note: You may want to create a `requirements.txt` file, but standard requirements generally include `pandas`, `requests`, `scikit-learn`, and `numpy`).
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Data Preprocessing
To clean the source data and fetch necessary metadata from DOI/OpenAlex, run the preprocessing script. This will read the files from the `src/` directory and output the cleaned datasets into the `modified_csv/` directory.

```bash
python main_s2_v2.py <add_your_actions_here>
```
Actions:
- -pc: 'precrawl'
- -c: 'crawl'
- -p: 'preprocess'
- -t: 'train'
- -s: 'submit'
