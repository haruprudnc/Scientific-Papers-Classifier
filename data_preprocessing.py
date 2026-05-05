import argparse
import re
import pandas as pd
import numpy as np
import joblib

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sentence_transformers import SentenceTransformer
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

import pandas as pd
import numpy as np
import joblib
import os

# --- Classifiers ---
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

# Assuming these are your local custom modules
from utils.terminal import get_terminal_length
from utils.color import Color
from utils.cli_progress_bar import cliProgressBar
from core.doi_org import DoiOrg
from core.openalex_org import OpenAlex

# File Paths
TRAIN_SOURCE_PATH = "src/Stage_1_publcitrain.csv"
TRAIN_RAW_DATA_PATH = "./modified_csv/train_raw_data.csv"
TRAIN_PREPROCESS_PATH = "./modified_csv/train_preprocess_data.csv"


TEST_SOURCE_PATH = "src/test_(2).csv"
TEST_RAW_DATA_PATH = "./modified_csv/test_raw_data.csv"
TEST_PREPROCESS_PATH = "./modified_csv/test_preprocess_data.csv"


MODEL_PATH = "./models/model.pkl"


SAVE_COLS_AFTER_SCRAPE      = ["id", "title", "title_doiorg", "title_openalex", "authors", "authors_doiorg", "authors_openalex", "authors_count_doiorg", "authors_count_openalex", "doi", "venue", "year", "abstract_doiorg", "abstract_openalex", "primary_topic", "keywords", "concepts"]
SAVE_COLS_AFTER_PREPROCESS  = ['title', 'authors', 'authors_count_token', 'authors_count_numberic', 'abstracts', 'venue', 'year_token', 'year', 'primary_topic', 'keywords']


TEXT_FEATURES       = ['title', 'abstracts', 'keywords', 'venue']
NUMBERIC_FEATURES   = []

# .-. ======================================== SCRAPING DATA ======================================== .-. #
def scrape_data(df, target = None, testing = False):
    # df["authors_doiorg"] = ""
    # df["authors_openalex"] = ""
    # df["title_doiorg"] = ""
    # df["title_openalex"] = ""
    # df["primary_topic"] = ""
    # df["keywords"] = ""
    # df["concepts"] = ""
    # df["abstract_doiorg"] = ""
    # df["abstract_openalex"] = ""

    total_rows = len(df.index)
    new_data = []
    for i, row in enumerate(df.itertuples(index = True), start = 1):
        if i > 2 and testing:
            break

        try:
            doiorg = DoiOrg(row.doi)
            openalex = OpenAlex(row.doi)

            new_data.append({
                "Index"                 : row.Index,
                "authors_doiorg"        : str(doiorg.authors),
                "authors_count_doiorg"  : int(len(doiorg.authors)),
                "title_doiorg"          : doiorg.title,
                "abstract_doiorg"       : doiorg.abstract,
                "authors_openalex"      : str(openalex.authors),
                "authors_count_openalex": int(len(openalex.authors)),
                "title_openalex"        : openalex.title,
                "abstract_openalex"     : openalex.abstract,
                "primary_topic"         : openalex.primary_topic,
                "keywords"              : str(openalex.keywords),
                "concepts"              : str(openalex.concepts)
            })

            cliProgressBar(i, total_rows, 'Scraping Data >v<')

        except Exception as e:
            print('Error at doi: ', row.doi, e)

    if new_data:
        new_df = pd.DataFrame(new_data).set_index('Index')
        raw_df = df.join(new_df)

        raw_df['authors_count_doiorg'] = raw_df['authors_count_doiorg'].astype('Int64')
        raw_df['authors_count_openalex'] = raw_df['authors_count_openalex'].astype('Int64')

    if target:
        sorted_df = raw_df[SAVE_COLS_AFTER_SCRAPE + ["Label"]]
        sorted_df.to_csv(TRAIN_RAW_DATA_PATH, index = False)
    else:
        sorted_df = raw_df[SAVE_COLS_AFTER_SCRAPE]
        sorted_df.to_csv(TEST_RAW_DATA_PATH, index = False)

    return sorted_df


# .-. ======================================== PREPROCESSING DATA ======================================== .-. #
def clean_text(text):
    return re.sub(r'[^a-zA-Z0-9\s]', '', str(text))

def clean_fillers(text):
    # nltk.download('punkt')
    # nltk.download('punkt_tab')
    # nltk.download('stopwords')
    
    words = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word not in stop_words]
    clean_txt = " ".join(filtered_words)
    return clean_txt

def preprocess_title(string: str):
    new_text = str(string).lower()
    clean_txt = clean_text(new_text)
    no_filler = clean_fillers(clean_txt)
    return no_filler

def preprocess_abstracts(string: str):
    new_text = str(string).lower()
    clean_txt = clean_text(new_text)
    no_filler = clean_fillers(clean_txt)
    return no_filler

def preprocss_authors(string: str):
    if pd.isna(string):
        return ""
    string = str(string).lower()
    authors = [" ".join(author.split()) for author in string.split(",") if author.strip()]
    return " ".join(authors)

def preprocess_authors_count(count):
    if count <= 0:
        return "no_author"
    if count == 1:
        return "single_author"
    if count <= 3:
        return "few_authors"
    return "many_authors"

def preprocess_year(year):
    if year < 2016:
        return "year_pre_2016"
    if year < 2018:
        return "year_2016_2017"
    if year < 2020:
        return "year_2018_2019"
    if year < 2022:
        return "year_2020_2021"
    if year < 2024:
        return "year_2022_2023"
    return "year_post_2024"

def preprocess_data(df: pd.DataFrame, target = None):
    # >-< --------------------------------- HANDLING TITLE --------------------------------- >-< #
    df['title'] = df['title'].fillna(df['title_doiorg'].fillna(df['title_openalex']))
    df['title'] = df['title'].apply(preprocess_title)

    # >-< --------------------------------- HANDLING ABSTRACTS --------------------------------- >-< #
    df['abstracts'] = df['abstract_openalex'].fillna(df['abstract_doiorg'])
    df['abstracts'] = df['abstracts'].apply(preprocess_abstracts)

    # >-< --------------------------------- HANDLING AUTHORS --------------------------------- >-< #
    df['authors'] = df['authors'].fillna(df['authors_doiorg'].fillna(df['authors_openalex']))
    df['authors'] = df['authors'].apply(preprocss_authors)

    # >-< --------------------------------- HANDLING AUTHORS COUNT --------------------------------- >-< #
    df['authors_count_numberic'] = df['authors_count_doiorg'].fillna(df['authors_count_openalex'])
    df['authors_count_token'] = df['authors_count_numberic'].apply(preprocess_authors_count)

    # >-< --------------------------------- HANDLING YEAR TOKEN --------------------------------- >-< #
    df['year_token'] = df['year'].apply(preprocess_year)

    # >-< --------------------------------- HANDLING TOPIC & KEYWORDS --------------------------------- >-< #
    df['primary_topic'] = df['primary_topic'].apply(preprocess_abstracts)
    df['keywords'] = df['keywords'].apply(preprocess_abstracts)

    # >-< --------------------------------- SCALING YEAR & AUTHOR COUNT --------------------------------- >-< #
    scaler = StandardScaler()
    df[['year', 'authors_count_numberic']] = scaler.fit_transform(df[['year', 'authors_count_numberic']])

    if target:
        save_df = df[SAVE_COLS_AFTER_PREPROCESS + [target]]
        save_df.to_csv(TRAIN_PREPROCESS_PATH, index = False)
    else:
        save_df = df[SAVE_COLS_AFTER_PREPROCESS + ['id']]
        save_df.to_csv(TEST_PREPROCESS_PATH, index = False)
    return save_df


# .-. ======================================== TRAINING MODEL ======================================== .-. #
def train_model():
    print(f"Loading preprocessed training data...")
    # Assuming TRAIN_PREPROCESS_PATH is defined globally
    df = pd.read_csv(TRAIN_PREPROCESS_PATH)

    # 1. Handle missing values
    text_cols = ['title', 'abstracts', 'keywords', 'primary_topic', 'venue']
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna("")

    # 2. Combine text for Vectorization
    print(f"Encoding text features using SentenceTransformer...")
    df['combined_text'] = df[TEXT_FEATURES].fillna('').agg(' '.join, axis=1)

    embedder = SentenceTransformer('allenai/specter2_base')
    text_embeddings = embedder.encode(df['combined_text'].tolist(), show_progress_bar=True)

    # 3. Combine with numeric features
    numeric_features = df[NUMBERIC_FEATURES].fillna(0).values
    X_sklearn = np.hstack((text_embeddings, numeric_features))
    y = df['Label'].values

    # 4. Train/Test Split (stratified is recommended for classification)
    X_train_sk, X_val_sk, y_train, y_val = train_test_split(
        X_sklearn, y, test_size=0.2, random_state=42, stratify=y
    )

    # ==========================================
    # ENSEMBLE CLASSIFIER (Sklearn + XGBoost)
    # ==========================================
    print(f"Initializing base models for Sklearn Voting Classifier...")
    
    xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    lr_model = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
    svm_model = SVC(probability=True, random_state=42) # probability=True required for soft voting

    sklearn_voter = VotingClassifier(
        estimators=[
            ('xgb', xgb_model),
            ('rf', rf_model),
            ('gb', gb_model),
            ('lr', lr_model),
            ('svm', svm_model)
        ],
        voting='soft' 
    )

    print(f"Training Sklearn Voting Classifier...")
    sklearn_voter.fit(X_train_sk, y_train)

    # ==========================================
    # EVALUATION
    # ==========================================
    print(f"Evaluating Ensemble on validation set...")
    
    y_pred = sklearn_voter.predict(X_val_sk)

    # Evaluate
    print(f"\nValidation Accuracy: {accuracy_score(y_val, y_pred):.4f}")
    print(f"Classification Report:")
    print(classification_report(y_val, y_pred))

    # Save models
    joblib.dump({
        'sklearn_voter': sklearn_voter, 
        'embedder_name': 'allenai/specter2_base'
    }, MODEL_PATH)
    
    print(f"Model successfully saved to sklearn_ensemble_model.pkl")


# .-. ======================================== PREDICTION ======================================== .-. #
def predict():
    print(f"{Color.blue}Loading preprocessed test data & models...{Color.reset}")
    # Assuming TEST_PREPROCESS_PATH is defined globally
    df = pd.read_csv(TEST_PREPROCESS_PATH)
    
    # --- Load Sklearn Ensemble model ---
    artifacts = joblib.load(MODEL_PATH)
    sklearn_voter = artifacts['sklearn_voter']
    embedder = SentenceTransformer(artifacts['embedder_name'])

    # 1. Handle missing values
    text_cols = ['title', 'abstracts', 'keywords', 'primary_topic', 'venue']
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna("")

    # 2. Combine text for Vectorization
    print(f"{Color.purple}Encoding test text features...{Color.reset}")
    # CRITICAL: This MUST match your training feature string perfectly. 
    # In the training script, 'keywords' was omitted, so we omit it here too.
    # df['combined_text'] = (df['title'] + " " + df['abstracts'] + " " + df['primary_topic'] + " " + df['venue'])
    df['combined_text'] = df[TEXT_FEATURES].fillna('').agg(' '.join, axis=1)
    
    text_embeddings = embedder.encode(df['combined_text'].tolist(), show_progress_bar=True)

    # 3. Combine with numeric features
    numeric_features = df[NUMBERIC_FEATURES].fillna(0).values
    X_test_sk = np.hstack((text_embeddings, numeric_features))

    # 4. Predict using Ensemble
    print(f"{Color.purple}Generating predictions...{Color.reset}")
    predictions = sklearn_voter.predict(X_test_sk)

    # 5. Save Submission
    submission = pd.DataFrame({
        'id': df['id'],
        'Label': predictions
    })
    submission.to_csv('submission.csv', index=False)
    print(f"{Color.green}{Color.bold}Predictions successfully saved to submission.csv{Color.reset}")

# .-. ======================================== MAIN PROGRAM ======================================== .-. #
if __name__ == "__main__":
    startup_msg = "    world.execute(me)    ".center(get_terminal_length(), "=")
    print(f"{Color.pink}{Color.bold}{startup_msg}{Color.reset}")

    # ^^ ======================================== PARSER ======================================== ^^ #
    parser = argparse.ArgumentParser(description="A simple argument parser.")

    parser.add_argument("--scrape", action = "store_true", help = "Default is False, use to start scraping data.")
    parser.add_argument("-p", "--preprocess", action = "store_true", help = "Default is False, use to start preprocessing data.")
    parser.add_argument("-t", "--train", action = "store_true", help = "Default is False, use to start training model.")
    parser.add_argument("-s", "--submit", action = "store_true", help = "Default is False, use to start predicting.")

    args = parser.parse_args()

    print(f"{Color.blue}{' CONFIG '.center(int(get_terminal_length()/2), 'v')}{Color.reset}")
    print(f"{Color.purple}Scrape: {f'{Color.green}True' if args.scrape else f'{Color.red}False'}{Color.reset}")
    print(f"{Color.purple}Preprocess: {f'{Color.green}True' if args.preprocess else f'{Color.red}False'}{Color.reset}")
    print(f"{Color.purple}Training: {f'{Color.green}True' if args.train else f'{Color.red}False'}{Color.reset}")
    print(f"{Color.purple}Predict: {f'{Color.green}True' if args.submit else f'{Color.red}False'}{Color.reset}")
    print("")

    # ^^ ======================================== IMPORT CONFIG ======================================== ^^ #
    # if args.submit:
    #     data_src_path = TEST_SOURCE_PATH
    #     raw_scrape_path = TEST_RAW_DATA_PATH
    #     target = None
    # else:
    #     data_src_path = TRAIN_SOURCE_PATH
    #     raw_scrape_path = TRAIN_RAW_DATA_PATH
    #     target = "Label"

    # ^^ ======================================== SCRAPING DATA ======================================== ^^ #
    if args.scrape:
        df = pd.read_csv(TRAIN_SOURCE_PATH)
        scrape_data(df, "Label")
        df = pd.read_csv(TEST_SOURCE_PATH)
        scrape_data(df)
        print(f"{Color.green}{Color.bold}{"    SCRAPING SUCCESS    ".center(get_terminal_length(), "=")}{Color.reset}")

    # ^^ ======================================== PREPROCESSING DATA ======================================== ^^ #
    if args.preprocess:
        df = pd.read_csv(TRAIN_RAW_DATA_PATH)
        preprocess_data(df, "Label")
        df = pd.read_csv(TEST_RAW_DATA_PATH)
        preprocess_data(df)
        print(f"{Color.green}{Color.bold}{"    PREPROCESSING SUCCESS    ".center(get_terminal_length(), "=")}{Color.reset}")


# ^^ ======================================== TRAINING MODEL ======================================== ^^ #
    if args.train:
        train_model()
        print(f"{Color.green}{Color.bold}{'    TRAINING SUCCESS    '.center(get_terminal_length(), '=')}{Color.reset}")
    
    # ^^ ======================================== PREDICTION ======================================== ^^ #
    if args.submit:
        predict()
        print(f"{Color.green}{Color.bold}{'    PREDICTION SUCCESS    '.center(get_terminal_length(), '=')}{Color.reset}")
    
    winddown_msg = "    world.terminate(me)    ".center(get_terminal_length(), "=")
    print(f"{Color.pink}{Color.bold}{winddown_msg}{Color.reset}")