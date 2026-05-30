from dotenv import load_dotenv
import os

# Load variables from the .env file
load_dotenv()

SEMANTICSCHOLAR_API_KEY =  os.getenv("SEMANTICSCHOLAR_API_KEY")

# .-. FILE PATHS
TRAIN_SOURCE_PATH       = "./src/train.csv"
TRAIN_PRECRAWL_PATH     = "./modified_csv/stage2/train_precrawl_data.csv"
TRAIN_POSTCRAWL_PATH    = "./modified_csv/stage2/train_postcrawl_data.csv"
TRAIN_PREPROCESS_PATH   = "./modified_csv/stage2/train_preprocess_data.csv"


TEST_PUBLIC_SOURCE_PATH     = "./src/public_test.csv"
TEST_PUBLIC_PRECRAWL_PATH   = "./modified_csv/stage2/test_public_precrawl_data.csv"
TEST_PUBLIC_POSTCRAWL_PATH  = "./modified_csv/stage2/test_public_postcrawl_data.csv"
TEST_PUBLIC_PREPROCESS_PATH = "./modified_csv/stage2/test_public_preprocess_data.csv"

TEST_PRIVATE_SOURCE_PATH        = "./src/private_test.csv"
TEST_PRIVATE_PRECRAWL_PATH      = "./modified_csv/stage2/test_private_precrawl_data.csv"
TEST_PRIVATE_POSTCRAWL_PATH     = "./modified_csv/stage2/test_private_postcrawl_data.csv"
TEST_PRIVATE_PREPROCESS_PATH    = "./modified_csv/stage2/test_private_preprocess_data.csv"


SUBMISSION_PUBLIC_PATH  = "./submissions/stage2/public_submission.csv"
SUBMISSION_PRIVATE_PATH = "./submissions/stage2/private_submission.csv"
COMBINE_SUBMISSION_PATH = "./submissions/stage2/combine_public_private_submission.csv"


# .-. PRECRAWL & CRAWL
SAVED_COLS_AFTER_CRAWL      = ["id", "title", "title_doiorg", "title_openalex", "authors", "authors_doiorg", "authors_openalex", "authors_count_doiorg", "authors_count_openalex", "venue", "year", "abstract_doiorg", "abstract_openalex", "abstract_semantic", "tldr", "primary_topic", "keywords", "concepts", "reference_count", "citation_count", "is_open_access"]

# .-. PREPROCESS
# LOWERCASE = False
CLEAN_PUNCTUATION = False
CLEAN_FILLERS = False
LEMMATIZE = False

LOWERCASE = True
# CLEAN_PUNCTUATION = True
# CLEAN_FILLERS = True
# LEMMATIZE = True

SAVED_COLS_AFTER_PREPROCESS  = ['title', 'authors', 'abstracts', 'venue', 'year_token', 'year', 'primary_topic', "reference_count", "citation_count", 'keywords']

# .-. TRAIN MODEL
# TEXT_FEATURES       = ['title', 'abstracts', 'keywords']
# TEXT_FEATURES       = ['title', 'abstracts']
TEXT_FEATURES       = ['title']
# NUMBERIC_FEATURES   = ['year']
NUMBERIC_FEATURES   = ['year', 'reference_count']

EMBEDDING_MODEL = "allenai/specter2_base"
MODEL_PATH = "./models/stage2_model.pkl"