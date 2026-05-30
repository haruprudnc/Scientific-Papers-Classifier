import pandas as pd
import numpy as np
import time
import requests

from config import config
from core.doi_org import DoiOrg
from core.openalex_org import OpenAlex
from core.semanticscholar_org import SemanticScholar, SemanticScholarDOI
from utils.cli_progress_bar import cliProgressBar
from utils.color import Color


# .-. ======================================== PRECRAWLING DATA ======================================== .-. #
def is_semantics(text) -> bool:
    if not isinstance(text, str):
        return False
    base = "https://www.semanticscholar.org/paper/"
    if text.startswith(base):
        return True
    return False

def is_doi(text) -> bool:
    if not isinstance(text, str):
        return False
    if text.startswith("10"):
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

def precrawl_process(df: pd.DataFrame, save_path):
    df['semantics_id'] = df['doi'].apply(sort_semantics)
    df['other'] = df['doi'].apply(lambda x: x if not (is_semantics(x) or is_doi(x)) else np.nan)
    df['doi'] = df['doi'].apply(lambda x: x if is_doi(x) else np.nan)
    
    df.to_csv(save_path, index=False)
    return df


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

            if pd.isna(row.doi) and not pd.isna(row.semantics_id):
                semantics = SemanticScholar(row.semantics_id)
            else:
                semantics = SemanticScholarDOI(doi)

            attempt = 0

            while True:
                try:
                    # Sleep longer on subsequent retries to avoid rate limits
                    time.sleep(1 + (attempt * 2)) 
                    semantics.get_details()
                    doi = semantics.doi
                    break  # If successful, break out of the retry loop
                except requests.exceptions.HTTPError as inner_e:
                    status_code = inner_e.response.status_code
                
                    if status_code == 404:
                        print("SemanticScholar Error 404: Paper Not Found. Stopping.")
                        break  # Stop retrying if the resource doesn't exist
                        
                    elif status_code == 429:
                        attempt += 1
                        print(f"SemanticScholar Error 429: Rate limited. Retrying (Attempt {attempt})...")
                        continue  # Keep looping indefinitely on 429
                        
                    else:
                        # If it's a 401, 500, etc., stop and raise the error
                        print(f"SemanticScholar failed with HTTP {status_code}: {inner_e}")
                        break
                        # raise inner_e

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
                "tldr"                  : semantics.tldr,
                "reference_count"       : semantics.reference_count,
                "citation_count"        : semantics.citation_count,
                "is_open_access"        : semantics.is_open_access
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
        raw_df['reference_count'] = raw_df['reference_count'].astype('Int64')
        raw_df['citation_count'] = raw_df['citation_count'].astype('Int64')

    if target:
        sorted_df = raw_df[config.saved_cols_after_crawl + [target]]
        sorted_df.to_csv(save_path, index = False)
    else:
        sorted_df = raw_df[config.saved_cols_after_crawl]
        sorted_df.to_csv(save_path, index = False)

    print(f"{Color.green}{Color.bold}Predictions successfully saved to {save_path}{Color.reset}")

    return sorted_df