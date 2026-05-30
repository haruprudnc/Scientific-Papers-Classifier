# Standard Library
import argparse
import re
import time

# Third-Party Libraries
import numpy as np
import pandas as pd

import joblib
from sentence_transformers import SentenceTransformer

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

# Local Custom Modules
from core.doi_org import DoiOrg
from core.openalex_org import OpenAlex
from core.semanticscholar_org import SemanticScholar
from utils.cli_progress_bar import cliProgressBar
from utils.color import Color
from utils.terminal import get_terminal_length



# File Paths
TRAIN_SOURCE_PATH       = "./src/train.csv"
TRAIN_PRECRAWL_PATH     = "./modified_csv/stage2/train_precrawl_data.csv"
TRAIN_POSTCRAWL_PATH     = "./modified_csv/stage2/train_postcrawl_data.csv"
TRAIN_PREPROCESS_PATH   = "./modified_csv/stage2/train_preprocess_data.csv"


TEST_PUBLIC_SOURCE_PATH        = "src/public_test.csv"
TEST_PUBLIC_PRECRAWL_PATH      = "./modified_csv/stage2/test_public_precrawl_data.csv"
TEST_PUBLIC_POSTCRAWL_PATH      = "./modified_csv/stage2/test_public_postcrawl_data.csv"
TEST_PUBLIC_PREPROCESS_PATH    = "./modified_csv/stage2/test_public_preprocess_data.csv"

SUBMISSION_PUBLIC_PATH         = "./submissions/stage2/public_submission.csv"


TEST_PRIVATE_SOURCE_PATH        = "src/private_test.csv"
TEST_PRIVATE_PRECRAWL_PATH      = "./modified_csv/stage2/test_private_precrawl_data.csv"
TEST_PRIVATE_POSTCRAWL_PATH      = "./modified_csv/stage2/test_private_postcrawl_data.csv"
TEST_PRIVATE_PREPROCESS_PATH    = "./modified_csv/stage2/test_private_preprocess_data.csv"

SUBMISSION_PRIVATE_PATH         = "./submissions/stage2/private_submission.csv"

MODEL_PATH = "./models/stage2_model.pkl"
COMBINE_SUBMISSION_PATH = "./submissions/stage2/combine_public_private_submission.csv"


SAVE_COLS_AFTER_CRAWL      = ["id", "title", "title_doiorg", "title_openalex", "authors", "authors_doiorg", "authors_openalex", "authors_count_doiorg", "authors_count_openalex", "doi", "venue", "year", "abstract_doiorg", "abstract_openalex", "abstract_semantic", "tldr", "primary_topic", "keywords", "concepts"]
SAVE_COLS_AFTER_PREPROCESS  = ['title', 'authors', 'authors_count_token', 'authors_count_numberic', 'abstracts', 'venue', 'year_token', 'year', 'primary_topic', 'keywords']


DO_LEMMATIZE = False

TEXT_FEATURES       = ['title', 'abstracts', 'keywords']
TEXT_FEATURES       = ['title', 'abstracts']
# TEXT_FEATURES       = ['title']
NUMBERIC_FEATURES   = ['year']


# .-. ======================================== PRECRAWLING DATA ======================================== .-. #
def is_semantics(text) -> bool:
    base = "https://www.semanticscholar.org/paper/"
    if text[0:len(base)] == base:
        return True
    return False

def is_doi(text) -> bool:
    if text[0:2] == "10":
        return True
    return False

def sort_semantics(text):
    base = "https://www.semanticscholar.org/paper/"
    if is_semantics(text):
        return text[len(base):]
    return np.nan

def sort_doi(text):
    if is_doi(text):
        return text
    return np.nan

def get_doi_from_sematics(paper_id):
    print(paper_id)
    if pd.isna(paper_id):
        return paper_id
    
    try:
        semantics = SemanticScholar(paper_id, "s2k-9jEQxInLlfpQUGzB56OPUTJROA8l2e544yqnAgrF")
        print(semantics.doi)
        print(semantics.tldr)
        print(semantics.abstract)
        return semantics.doi
    except Exception as e:
        print(e)
        return np.nan
    finally:
        time.sleep(1.2)


def precrawl_process(df: pd.DataFrame, save_path):
    df['semantics_id'] = df['doi'].apply(sort_semantics)
    df['other'] = df['doi'].apply(lambda x: x if not (is_semantics(x) or is_doi(x)) else np.nan)
    df['doi'] = df['doi'].apply(lambda x: x if is_doi(x) else np.nan)

    # for i, row in enumerate
    # df['doi'] = df['semantics_id'].apply(get_doi_from_sematics).fillna(df['doi'])

    df.to_csv(save_path, index=False)
    return df


# print(sort_semantics("https://www.semanticscholar.org/paper/2e87df6f0d41a74b4fb37e0cfc755df5d4a27cc8"))
# print(sort_semantics("10.1145/2933575.2935310"))
# print(sort_doi("https://www.semanticscholar.org/paper/2e87df6f0d41a74b4fb37e0cfc755df5d4a27cc8"))
# print(sort_doi("10.1145/2933575.2935310"))

# df = pd.read_csv(TRAIN_SOURCE_PATH)
# preprocess_before_crawl(df)



# .-. ======================================== CRAWLING DATA ======================================== .-. #
def crawl_data(df, save_path, target = None, testing = False):
    # df["authors_doiorg"] = ""
    # df["authors_openalex"] = ""
    # df["title_doiorg"] = ""
    # df["title_openalex"] = ""
    # df["primary_topic"] = ""
    # df["keywords"] = ""
    # df["concepts"] = ""
    # df["abstract_doiorg"] = ""
    # df["abstract_openalex"] = ""
    
    # testing = True

    total_rows = len(df.index)
    new_data = []
    for i, row in enumerate(df.itertuples(index = True), start = 1):
        if i > 20 and testing:
            break

        try:
            doi = row.doi

            semantics = SemanticScholar(row.semantics_id, "s2k-9jEQxInLlfpQUGzB56OPUTJROA8l2e544yqnAgrF")
            if pd.isna(row.doi) and not pd.isna(row.semantics_id):
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        # Sleep longer on subsequent retries to avoid rate limits
                        time.sleep(1 + (attempt * 2)) 
                        semantics.get_details()
                        doi = semantics.doi
                        break  # If successful, break out of the retry loop
                    except Exception as inner_e:
                        print(f"SemanticScholar retry {attempt + 1}/{max_retries} failed: {inner_e}")
                        if attempt == max_retries - 1:
                            # If it's the last attempt, raise the error to be caught by the outer try-except
                            raise inner_e

            doiorg = DoiOrg(doi)
            openalex = OpenAlex(doi)

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
                "concepts"              : str(openalex.concepts),
                "abstract_semantic"     : semantics.abstract,
                "tldr"                  : semantics.tldr
            })

            # new_data.append({
            #     "Index"                 : row.Index,
            #     "authors_doiorg"        : "",
            #     "authors_count_doiorg"  : 0,
            #     "title_doiorg"          : "",
            #     "abstract_doiorg"       : "",
            #     "authors_openalex"      : "",
            #     "authors_count_openalex": 0,
            #     "title_openalex"        : "",
            #     "abstract_openalex"     : "",
            #     "primary_topic"         : "",
            #     "keywords"              : "",
            #     "concepts"              : ""
            # })

            cliProgressBar(i, total_rows, 'Crawling Data >v<')

        except Exception as e:
            print('Error at doi: ', row.doi, e)

    if new_data:
        new_df = pd.DataFrame(new_data).set_index('Index')
        raw_df = df.join(new_df)

        raw_df['authors_count_doiorg'] = raw_df['authors_count_doiorg'].astype('Int64')
        raw_df['authors_count_openalex'] = raw_df['authors_count_openalex'].astype('Int64')

    if target:
        sorted_df = raw_df[SAVE_COLS_AFTER_CRAWL + ["Label"]]
        sorted_df.to_csv(save_path, index = False)
    else:
        sorted_df = raw_df[SAVE_COLS_AFTER_CRAWL]
        sorted_df.to_csv(save_path, index = False)

    print(f"{Color.green}{Color.bold}Predictions successfully saved to {save_path}{Color.reset}")

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

def lemmatize_sentence(sentence):
    """
    Takes a string, processes it through spaCy, 
    and returns a string of lemmas.
    """
    if not DO_LEMMATIZE:
        return sentence

    # Load the English model (run 'python -m spacy download en_core_web_sm' in terminal first)
    nlp = spacy.load("en_core_web_sm")

    # Create a doc object
    doc = nlp(sentence)
    
    # Extract the lemma for each token
    lemmas = [token.lemma_ for token in doc]
    
    # Join them back into a single string
    return " ".join(lemmas)

def preprocess_title(string: str):
    new_text = str(string).lower()
    clean_txt = clean_text(new_text)
    no_filler = clean_fillers(clean_txt)
    lemma_txt = lemmatize_sentence(no_filler)
    return lemma_txt

def preprocess_abstracts(string: str):
    new_text = str(string).lower()
    clean_txt = clean_text(new_text)
    no_filler = clean_fillers(clean_txt)
    lemma_txt = lemmatize_sentence(no_filler)
    return lemma_txt

def preprocss_authors(string: str):
    if pd.isna(string):
        return ""
    string = str(string).lower()
    authors = [" ".join(author.split()) for author in string.split(",") if author.strip()]
    return " ".join(authors)

def preprocess_authors_count(count):
    if count <= 0:
        return "noauthor"
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


def preprocess_data(df: pd.DataFrame, save_path, target = None):
    # >-< --------------------------------- HANDLING TITLE --------------------------------- >-< #
    df['title'] = df['title'].fillna(df['title_doiorg'].fillna(df['title_openalex'].fillna("notitle")))
    df['title'] = df['title'].apply(preprocess_title)

    # >-< --------------------------------- HANDLING ABSTRACTS --------------------------------- >-< #
    df['abstracts'] = df['abstract_openalex'].fillna(df['abstract_doiorg'].fillna(df['abstract_semantic'].fillna(df['tldr'].fillna("noabstract"))))
    df['abstracts'] = df['abstracts'].apply(preprocess_abstracts)

    # >-< --------------------------------- HANDLING AUTHORS --------------------------------- >-< #
    df['authors'] = df['authors'].fillna(df['authors_doiorg'].fillna(df['authors_openalex'].fillna("noauthor")))
    df['authors'] = df['authors'].apply(preprocss_authors)

    # >-< --------------------------------- HANDLING AUTHORS COUNT --------------------------------- >-< #
    df['authors_count_numberic'] = df['authors_count_doiorg'].fillna(df['authors_count_openalex'].fillna(0))
    df['authors_count_token'] = df['authors_count_numberic'].apply(preprocess_authors_count)

    # >-< --------------------------------- HANDLING YEAR TOKEN --------------------------------- >-< #
    df['year_token'] = df['year'].apply(preprocess_year)

    # >-< --------------------------------- HANDLING TOPIC & KEYWORDS --------------------------------- >-< #
    df['primary_topic'] = df['primary_topic'].fillna("notopic").apply(preprocess_abstracts)
    df['keywords'] = df['keywords'].fillna("nokeywords").apply(preprocess_abstracts)

    # >-< --------------------------------- SCALING YEAR & AUTHOR COUNT --------------------------------- >-< #
    # scaler = StandardScaler()
    # df[['year', 'authors_count_numberic']] = scaler.fit_transform(df[['year', 'authors_count_numberic']])

    if target:
        save_df = df[SAVE_COLS_AFTER_PREPROCESS + [target]]
        save_df.to_csv(save_path, index = False)
    else:
        save_df = df[SAVE_COLS_AFTER_PREPROCESS + ['id']]
        save_df.to_csv(save_path, index = False)
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

    tfidf = TfidfVectorizer(max_features=500)
    tfidf_features = tfidf.fit_transform(df['venue']).toarray()

    # 3. Combine with numeric features
    numeric_features_raw = df[NUMBERIC_FEATURES].fillna(0).values
    scaler = StandardScaler()
    numeric_features_scaled = scaler.fit_transform(numeric_features_raw)

    X_sklearn = np.hstack((text_embeddings, numeric_features_scaled, tfidf_features))
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
    rf_model = RandomForestClassifier(n_estimators=1000, random_state=42, n_jobs=-1)
    gb_model = GradientBoostingClassifier(n_estimators=1000, random_state=42)
    lr_model = LogisticRegression(max_iter=10000, random_state=42, n_jobs=-1)
    svm_model = SVC(probability=True, random_state=42)

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
        'embedder_name': 'allenai/specter2_base',
        'scaler': scaler,
        'tfidf': tfidf
    }, MODEL_PATH)
    
    print(f"Model successfully saved to {MODEL_PATH}")


# .-. ======================================== PREDICTION ======================================== .-. #
def predict(test_preprocess_path, submission_path):
    print(f"{Color.blue}Loading preprocessed test data & models...{Color.reset}")
    # Assuming TEST_PREPROCESS_PATH is defined globally
    df = pd.read_csv(test_preprocess_path)
    
    # --- Load Sklearn Ensemble model ---
    artifacts = joblib.load(MODEL_PATH)
    sklearn_voter = artifacts['sklearn_voter']
    embedder = SentenceTransformer(artifacts['embedder_name'])
    scaler = artifacts['scaler']
    tfidf = artifacts['tfidf']

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

    tfidf_features = tfidf.transform(df['venue']).toarray()

    # 3. Combine with numeric features
    numeric_features_raw = df[NUMBERIC_FEATURES].fillna(0).values
    numeric_features_scaled = scaler.transform(numeric_features_raw)

    X_test_sk = np.hstack((text_embeddings, numeric_features_scaled, tfidf_features))

    # 4. Predict using Ensemble
    print(f"{Color.purple}Generating predictions...{Color.reset}")
    predictions = sklearn_voter.predict(X_test_sk)

    # 5. Save Submission
    submission = pd.DataFrame({
        'id': df['id'],
        'Label': predictions
    })
    submission.to_csv(submission_path, index=False)
    print(f"{Color.green}{Color.bold}Predictions successfully saved to {submission_path}{Color.reset}")
    return submission


# .-. ======================================== MAIN PROGRAM ======================================== .-. #
if __name__ == "__main__":
    startup_msg = "    world.execute(me)    ".center(get_terminal_length(), "=")
    print(f"{Color.pink}{Color.bold}{startup_msg}{Color.reset}")

    # ^^ ======================================== PARSER ======================================== ^^ #
    parser = argparse.ArgumentParser(description="A simple argument parser.")

    parser.add_argument("-pc", "--precrawl", action = "store_true", help = "Default is False, use to start precrawl process data.")
    parser.add_argument("-c", "--crawl", action = "store_true", help = "Default is False, use to start crawling data.")
    parser.add_argument("-p", "--preprocess", action = "store_true", help = "Default is False, use to start preprocessing data.")
    parser.add_argument("-t", "--train", action = "store_true", help = "Default is False, use to start training model.")
    parser.add_argument("-s", "--submit", action = "store_true", help = "Default is False, use to start predicting.")

    args = parser.parse_args()

    print(f"{Color.blue}{' CONFIG '.center(int(get_terminal_length()), 'v')}{Color.reset}")
    print(f"{Color.purple}Precrawl: {f'{Color.green}True' if args.precrawl else f'{Color.red}False'}{Color.reset}")
    print(f"{Color.purple}Crawl: {f'{Color.green}True' if args.crawl else f'{Color.red}False'}{Color.reset}")
    print(f"{Color.purple}Preprocess: {f'{Color.green}True' if args.preprocess else f'{Color.red}False'}{Color.reset}")
    print(f"{Color.purple}Training: {f'{Color.green}True' if args.train else f'{Color.red}False'}{Color.reset}")
    print(f"{Color.purple}Predict: {f'{Color.green}True' if args.submit else f'{Color.red}False'}{Color.reset}")
    print(f"{Color.blue}{'^'.center(int(get_terminal_length()), '^')}{Color.reset}")
    print("")

    # ^^ ======================================== IMPORT CONFIG ======================================== ^^ #
    # if args.submit:
    #     data_src_path = TEST_SOURCE_PATH
    #     raw_crawl_path = TEST_RAW_DATA_PATH
    #     target = None
    # else:
    #     data_src_path = TRAIN_SOURCE_PATH
    #     raw_crawl_path = TRAIN_RAW_DATA_PATH
    #     target = "Label"

    # ^^ ======================================== PRECRAWL PROCESSING ======================================== ^^ #
    if args.precrawl:
        df = pd.read_csv(TRAIN_SOURCE_PATH)
        df = precrawl_process(df, save_path=TRAIN_PRECRAWL_PATH)

        df = pd.read_csv(TEST_PUBLIC_SOURCE_PATH)
        df = precrawl_process(df, TEST_PUBLIC_PRECRAWL_PATH)

        df = pd.read_csv(TEST_PRIVATE_SOURCE_PATH)
        df = precrawl_process(df, TEST_PRIVATE_PRECRAWL_PATH)
        
        print(f"{Color.green}{Color.bold}{"    PRECRAWLING SUCCESS    ".center(get_terminal_length(), "=")}{Color.reset}")

    # ^^ ======================================== CRAWLING DATA ======================================== ^^ #
    if args.crawl:
        df = pd.read_csv(TRAIN_PRECRAWL_PATH)
        crawl_data(df, save_path=TRAIN_POSTCRAWL_PATH, target="Label")

        df = pd.read_csv(TEST_PUBLIC_PRECRAWL_PATH)
        crawl_data(df, save_path=TEST_PUBLIC_POSTCRAWL_PATH)

        df = pd.read_csv(TEST_PRIVATE_PRECRAWL_PATH)
        crawl_data(df, save_path=TEST_PRIVATE_POSTCRAWL_PATH)

        print(f"{Color.green}{Color.bold}{"    CRAWLING SUCCESS    ".center(get_terminal_length(), "=")}{Color.reset}")

    # ^^ ======================================== PREPROCESSING DATA ======================================== ^^ #
    if args.preprocess:
        import nltk
        import spacy
        from nltk.corpus import stopwords
        from nltk.tokenize import word_tokenize
        df = pd.read_csv(TRAIN_POSTCRAWL_PATH)
        preprocess_data(df, save_path=TRAIN_PREPROCESS_PATH, target="Label")
        df = pd.read_csv(TEST_PUBLIC_POSTCRAWL_PATH)
        preprocess_data(df, save_path=TEST_PUBLIC_PREPROCESS_PATH)
        df = pd.read_csv(TEST_PRIVATE_POSTCRAWL_PATH)
        preprocess_data(df, save_path=TEST_PRIVATE_PREPROCESS_PATH)
        print(f"{Color.green}{Color.bold}{"    PREPROCESSING SUCCESS    ".center(get_terminal_length(), "=")}{Color.reset}")


    # ^^ ======================================== TRAINING MODEL ======================================== ^^ #
    if args.train:
        print(f"{Color.blue}{Color.bold}{'    START TRAINING    '.center(get_terminal_length(), '=')}{Color.reset}")
        train_model()
        print(f"{Color.green}{Color.bold}{'    TRAINING SUCCESS    '.center(get_terminal_length(), '=')}{Color.reset}")
    
    # ^^ ======================================== PREDICTION ======================================== ^^ #
    if args.submit:
        print(f"{Color.blue}{Color.bold}{'    START PREDICTING    '.center(get_terminal_length(), '=')}{Color.reset}")
        public_summission = predict(TEST_PUBLIC_PREPROCESS_PATH, SUBMISSION_PUBLIC_PATH)
        private_summission = predict(TEST_PRIVATE_PREPROCESS_PATH, SUBMISSION_PRIVATE_PATH)
        combine = pd.concat([public_summission, private_summission])
        combine.to_csv(COMBINE_SUBMISSION_PATH, index=False)
        print(f"{Color.green}{Color.bold}{'    PREDICTION SUCCESS    '.center(get_terminal_length(), '=')}{Color.reset}")
    
    winddown_msg = "    world.terminate(me)    ".center(get_terminal_length(), "=")
    print(f"{Color.pink}{Color.bold}{winddown_msg}{Color.reset}")