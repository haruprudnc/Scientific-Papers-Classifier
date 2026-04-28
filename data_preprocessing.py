from utils.terminal import get_terminal_length
from utils.color import Color
from utils.cli_progress_bar import cliProgressBar
from core.doi_org import DoiOrg
from core.openalex_org import OpenAlex
import pandas as pd
import argparse
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# STARTING_FILE_PATH = 
TRAIN_SOURCE_PATH = "src/Stage_1_publcitrain.csv"
TRAIN_RAW_DATA_PATH = "./modified_csv/train_raw_data.csv"
TRAIN_PREPROCESS_PATH = "./modified_csv/train_preprocess_data.csv"


TEST_SOURCE_PATH = "src/test_(2).csv"
TEST_RAW_DATA_PATH = "./modified_csv/test_raw_data.csv"
TEST_PREPROCESS_PATH = "./modified_csv/test_preprocess_data.csv"


FEATURE_COLS = ['title', 'abstracts', 'primary_topic', 'venue']

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
        sorted_df = raw_df[["id", "title", "title_doiorg", "title_openalex", "authors", "authors_doiorg", "authors_openalex", "authors_count_doiorg", "authors_count_openalex", "doi", "venue", "year", "abstract_doiorg", "abstract_openalex", "primary_topic", "keywords", "concepts", "Label"]]
        sorted_df.to_csv(TRAIN_RAW_DATA_PATH, index = False)
    else:
        sorted_df = raw_df[["id", "title", "title_doiorg", "title_openalex", "authors", "authors_doiorg", "authors_openalex", "authors_count_doiorg", "authors_count_openalex", "doi", "venue", "year", "abstract_doiorg", "abstract_openalex", "primary_topic", "keywords", "concepts"]]
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

    # >-< --------------------------------- HANDLING YEAR --------------------------------- >-< #
    df['year_token'] = df['year'].apply(preprocess_year)

    df['primary_topic'] = df['primary_topic'].apply(preprocess_abstracts)
    df['keywords'] = df['keywords'].apply(preprocess_abstracts)

    if target:
        save_df = df[['title', 'authors', 'authors_count_token', 'abstracts', 'venue', 'year_token', 'primary_topic', 'keywords', 'Label']]
        save_df.to_csv(TRAIN_PREPROCESS_PATH, index = False)
    else:
        save_df = df[['title', 'authors', 'authors_count_token', 'abstracts', 'venue', 'year_token', 'primary_topic', 'keywords', 'id']]
        save_df.to_csv(TEST_PREPROCESS_PATH, index = False)
    return save_df


# .-. ======================================== SCRAPING DATA ======================================== .-. #
def train_model(df, target):
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report
    from sentence_transformers import SentenceTransformer
    import numpy as np
    import joblib

    print("--- Starting Transformer-based Model Training ---")
    
    # 1. Fill missing values with an empty string
    df = df.fillna("")
    
    # 2. Combine text features
    # For Transformers, feeding the text natively works best.
    # We use a separator [SEP] to logically divide the parts of the paper.
    df['combined_text'] = df[FEATURE_COLS].apply(lambda x: ' [SEP] '.join(x.astype(str)), axis=1)
    
    X_text = df['combined_text'].tolist()
    y = df[target].astype(int)
    
    # 3. Load the pre-trained Transformer Model
    print("Loading SentenceTransformer model (this might take a moment to download)...")
    # 'allenai/specter' is specifically trained on scientific papers!
    # If it is too heavy/slow, you can fall back to 'all-MiniLM-L6-v2'
    embedder = SentenceTransformer('allenai/specter') 
    
    # 4. Encode the text into dense vectors
    print("Encoding text into dense embeddings (this may take a few minutes)...")
    X_embeddings = embedder.encode(X_text, show_progress_bar=True)
    
    # 5. Split the embedded data
    X_train, X_test, y_train, y_test = train_test_split(
        X_embeddings, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 6. Train a classifier on top of the dense embeddings
    print("Training Logistic Regression classifier...")
    clf = LogisticRegression(
        C=2.0,                   # Mild regularization works well with dense embeddings
        class_weight='balanced', # Crucial for handling the imbalanced classes
        max_iter=1000, 
        solver='lbfgs',
        random_state=42
    )
    clf.fit(X_train, y_train)
    
    # 7. Evaluate and print results
    y_pred = clf.predict(X_test)
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred))
    
    # 8. Optionally save BOTH the classifier and the embedder for future use
    # joblib.dump(clf, 'classifier_head.pkl')
    # embedder.save('./specter_embedder')
    
    print("--- Model Training Completed ---")
    return clf, embedder


def predict(file_path, clf, embedder):
    df = pd.read_csv(file_path)
    df = df.fillna("")
    df['combined_text'] = df[FEATURE_COLS].apply(lambda x: ' [SEP] '.join(x.astype(str)), axis=1)
    X_text = df['combined_text'].tolist()
    X_embeddings = embedder.encode(X_text, show_progress_bar=True)
    y_pred = clf.predict(X_embeddings)

    new_df = pd.DataFrame({"id": df['id'], "Label": y_pred})
    new_df.to_csv("submition.csv", index = False)

    

# .-. ======================================== MAIN PROGRAME ======================================== .-. #
if __name__ == "__main__":
    startup_msg = "    world.execute(me)    ".center(get_terminal_length(), "=")
    print(f"{Color.pink}{Color.bold}{startup_msg}{Color.reset}")

    # ^^ ======================================== PARSER ======================================== ^^ #
    parser = argparse.ArgumentParser(description = "A simple argument parser.")

    parser.add_argument("-s", "--scrape", action = "store_true", help = "Default is False, use to start scraping data.")
    parser.add_argument("-prep", "--preprocess", action = "store_true", help = "Default is False, use to start preprocessing data.")
    parser.add_argument("--submit", action = "store_true")

    args = parser.parse_args()

    print(f"{Color.blue}{" CONFIG ".center(int(get_terminal_length()/2), "v")}{Color.reset}")
    print(f"{Color.purple}Scrape: {f"{Color.green}True" if args.scrape else f"{Color.red}False"}{Color.reset}")
    print(f"{Color.purple}Preprocess: {f"{Color.green}True" if args.preprocess else f"{Color.red}False"}{Color.reset}")
    # print(f"{Color.purple}Preprocess: {f"{Color.green}True" if args.preprocess else f"{Color.red}False"}{Color.reset}")
    print("")

    # ^^ ======================================== IMPORT CONFIG ======================================== ^^ #
    if args.submit:
        data_src_path = TEST_SOURCE_PATH
        raw_scrape_path = TEST_RAW_DATA_PATH
        target = None
    else:
        data_src_path = TRAIN_SOURCE_PATH
        raw_scrape_path = TRAIN_RAW_DATA_PATH
        target = "Label"

    # ^^ ======================================== SCRAPING DATA ======================================== ^^ #
    if args.scrape:
        df = pd.read_csv(data_src_path)
        scrape_data(df, target)

    # ^^ ======================================== PREPROCESSING DATA ======================================== ^^ #
    if args.preprocess:
        df = pd.read_csv(raw_scrape_path)
        preprocess_data(df, target)

    # ^^ ======================================== TRAINING MODEL ======================================== ^^ #
    df = pd.read_csv(TRAIN_PREPROCESS_PATH)
    clf, embedder = train_model(df, target)
    predict(TEST_PREPROCESS_PATH, clf, embedder)

    winddown_msg = "    world.terminate(me)    ".center(get_terminal_length(), "=")
    print(f"{Color.pink}{Color.bold}{winddown_msg}{Color.reset}")