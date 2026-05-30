import pandas as pd
import numpy as np
import joblib
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, cohen_kappa_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel

from config import config 


# .-. ======================================== TRAINING MODEL ======================================== .-. #
def train_model_specter_classifier(train_path, model_path):
    # ^^ LOAD DATAFRAME
    print("Loading preprocessed training data...")
    df = pd.read_csv(train_path)

    # ^^ HANDLE MISSING DATA
    # text_cols = ['title', 'abstracts', 'keywords', 'primary_topic', 'venue']
    # for col in text_cols:
    #     if col in df.columns:
    #         df[col] = df[col].fillna("")

    # df = df.dropna(subset=['abstracts'])

    # ^^ SPECTER EMBEDDING 
    print(f"Encoding text features using SentenceTransformer...")
    
    embedder = SentenceTransformer(config.EMBEDDING_MODEL)

    sep_token = embedder.tokenizer.sep_token
    df['combined_text'] = df['title'] + sep_token + df['abstracts']

    text_embeddings = embedder.encode(df['combined_text'].tolist(), show_progress_bar=True)
    
    # ^^ TFIDF EMBEDDING
    tfidf = TfidfVectorizer(max_features=20)
    tfidf_features = tfidf.fit_transform(df['venue']).toarray()

    # ^^ SCALING NUMBERICAL
    numeric_features_raw = df[config.NUMBERIC_FEATURES].fillna(0).values
    scaler = StandardScaler()
    numeric_features_scaled = scaler.fit_transform(numeric_features_raw)

    # ^^ FINALIZE DATA FOR TRAINING
    X_sklearn = np.hstack((text_embeddings, numeric_features_scaled, tfidf_features))
    y = df['Label'].values

    X_train_sk, X_val_sk, y_train, y_val = train_test_split(
        X_sklearn, y, test_size=0.01, random_state=42, stratify=y
    )

    # ==========================================
    # ENSEMBLE CLASSIFIER (Sklearn + XGBoost)
    # ==========================================
    print(f"Initializing base models for Sklearn Voting Classifier...")
    
    xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    lr_model = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
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
    qwk_score = cohen_kappa_score(y_val, y_pred, weights='quadratic')
    print(f"QWK Score: {qwk_score:.4f}")
    print(f"Classification Report:")
    print(classification_report(y_val, y_pred))

    # Save models
    joblib.dump({
        'sklearn_voter': sklearn_voter, 
        'embedder_name': config.EMBEDDING_MODEL,
        'scaler': scaler,
        'tfidf': tfidf
    }, model_path)
    
    print(f"Model successfully saved to {model_path}")

def train_model(train_path, model_path):
    print("Loading preprocessed training data...")
    df = pd.read_csv(train_path)

    # 1. Handle missing values
    text_cols = ['title', 'abstracts', 'keywords', 'primary_topic', 'venue']
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna("")

    

    # 2. Combine text for Vectorization
    print(f"Encoding text features using SentenceTransformer...")
    
    embedder = SentenceTransformer(config.EMBEDDING_MODEL)

    sep_token = embedder.tokenizer.sep_token
    df['combined_text'] = df['title'] + sep_token + df['abstracts']

    text_embeddings = embedder.encode(df['combined_text'].tolist(), show_progress_bar=True)

    tfidf = TfidfVectorizer(max_features=500)
    tfidf_features = tfidf.fit_transform(df['venue']).toarray()

    # 3. Combine with numeric features
    numeric_features_raw = df[config.NUMBERIC_FEATURES].fillna(0).values
    scaler = StandardScaler()
    numeric_features_scaled = scaler.fit_transform(numeric_features_raw)

    
    X_sklearn = np.hstack((text_embeddings, numeric_features_scaled, tfidf_features))
    y = df['Label'].values

    # 4. Train/Test Split
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
        'embedder_name': config.EMBEDDING_MODEL,
        'scaler': scaler,
        'tfidf': tfidf
    }, model_path)
    
    print(f"Model successfully saved to {model_path}")