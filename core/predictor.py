import joblib
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

from utils.color import Color
from config import config

# .-. ======================================== PREDICTION ======================================== .-. #
def predict(test_preprocess_path, submission_path):
    # ^^ LOAD DATAFRAME
    print(f"{Color.blue}Loading preprocessed test data & models...{Color.reset}")
    df = pd.read_csv(test_preprocess_path)
    
    # ^^ LOAD ARTIFACTS
    artifacts = joblib.load(config.MODEL_PATH)
    sklearn_voter = artifacts['sklearn_voter']
    embedder = SentenceTransformer(artifacts['embedder_name'])
    scaler = artifacts['scaler']
    tfidf = artifacts['tfidf']

    # ^^ HANDLE MISSING DATA
    text_cols = ['title', 'abstracts', 'keywords', 'primary_topic', 'venue']
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna("")

    # ^^ SPECTER EMBEDDING 
    print(f"Encoding text features using SentenceTransformer...")
    
    sep_token = embedder.tokenizer.sep_token
    df['combined_text'] = df['title'] + sep_token + df['abstracts']
    
    text_embeddings = embedder.encode(df['combined_text'].tolist(), show_progress_bar=True)

    # ^^ TFIDF EMBEDDING
    tfidf_features = tfidf.transform(df['venue']).toarray()

    # ^^ SCALING NUMBERICAL
    numeric_features_raw = df[config.NUMBERIC_FEATURES].fillna(0).values
    numeric_features_scaled = scaler.transform(numeric_features_raw)

    # ^^ FINALIZE DATA FOR PREDICTING
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