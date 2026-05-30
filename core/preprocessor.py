import re
import nltk
import spacy
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# New imports for structured metadata preprocessing
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from config import config

def lowercase_text(text):
    if config.LOWERCASE:
        return text.lower()
    return text

def clean_punctuation(text):
    if config.CLEAN_PUNCTUATION:
        return re.sub(r'[^a-zA-Z0-9\s]', '', str(text))
    else:
        return text

def clean_fillers(text):
    # nltk.download('punkt')
    # nltk.download('punkt_tab')
    # nltk.download('stopwords')
    
    if config.CLEAN_FILLERS:
        words = word_tokenize(lowercase_text(text))
        stop_words = set(stopwords.words('english'))
        filtered_words = [word for word in words if word not in stop_words]
        clean_txt = " ".join(filtered_words)
        return clean_txt
    
    return text

def lemmatize_sentence(sentence):
    """
    Takes a string, processes it through spaCy, 
    and returns a string of lemmas.
    """
    if config.LEMMATIZE:
        # Load the English model (run 'python -m spacy download en_core_web_sm' in terminal first)
        nlp = spacy.load("en_core_web_sm")

        # Create a doc object
        doc = nlp(sentence)
        
        # Extract the lemma for each token
        lemmas = [token.lemma_ for token in doc]
        
        # Join them back into a single string
        return " ".join(lemmas)
    
    return sentence

def preprocess_title(text: str):
    new_text = lowercase_text(str(text))
    clean_txt = clean_punctuation(new_text)
    no_filler = clean_fillers(clean_txt)
    lemma_txt = lemmatize_sentence(no_filler)
    return lemma_txt

def preprocess_abstracts(text: str):
    new_text = lowercase_text(str(text))
    clean_txt = clean_punctuation(new_text)
    no_filler = clean_fillers(clean_txt)
    lemma_txt = lemmatize_sentence(no_filler)
    return lemma_txt

def preprocss_authors(text: str):
    if pd.isna(text):
        return ""
    
    new_text = lowercase_text(str(text))
    authors = [" ".join(author.split()) for author in new_text.split(",") if author.strip()]
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
    if pd.isna(year):
        return "year_unknown"
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


def preprocess_data(df: pd.DataFrame, save_path, target=None):
    # >-< --------------------------------- HANDLING TITLE --------------------------------- >-< #
    df['title'] = df['title'].fillna(df['title_doiorg'].fillna(df['title_openalex'].fillna("notitle")))
    df['title'] = df['title'].fillna(df['title_doiorg'].fillna(df['title_openalex'].fillna(np.nan)))
    df['title'] = df['title'].apply(preprocess_title)

    # >-< --------------------------------- HANDLING ABSTRACTS --------------------------------- >-< #
    df['abstracts'] = df['abstract_openalex'].fillna(df['abstract_doiorg'].fillna(df['abstract_semantic'].fillna("noabstract")))
    # df['abstracts'] = df['abstract_openalex'].fillna(df['abstract_doiorg'].fillna(df['abstract_semantic'].fillna(np.nan)))
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

    # >-< --------------------------------- HANDLING VENUE (ONE-HOT ENCODING) -------------- >-< #
    # 1. Handle missing venues
    df['venue'] = df['venue'].fillna("novenue")
    
    # 2. Initialize OneHotEncoder 
    # sparse_output=False returns a numpy array instead of a sparse matrix
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    
    # 3. Fit and transform
    venue_encoded = encoder.fit_transform(df[['venue']])
    
    # 4. Create feature names and convert back to DataFrame
    venue_cols = encoder.get_feature_names_out(['venue'])
    venue_df = pd.DataFrame(venue_encoded, columns=venue_cols, index=df.index)
    
    # 5. Concatenate with the main dataframe
    df = pd.concat([df, venue_df], axis=1)

    # >-< --------------------------------- SAVING DATA --------------------------------- >-< #
    # Dynamically add the new one-hot encoded venue columns to the saved columns list
    saved_cols = config.SAVED_COLS_AFTER_PREPROCESS + list(venue_cols)

    if target:
        save_df = df[saved_cols + [target]]
        save_df.to_csv(save_path, index=False)
    else:
        save_df = df[saved_cols + ['id']]
        save_df.to_csv(save_path, index=False)
        
    return save_df