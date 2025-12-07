"""
========================================================================================================================
RegPolicyBot – DATA MODELLING & EMBEDDINGS
------------------------------------------------------------------------------------------------------------------------
Purpose: In this script, I trained and evaluated several NLP models on the cleaned Regulatory Studies Center (RSC) corpus
         and prepared the artifacts that the RegPolicyBot chatbot will use. Concretely, I:

         - Loaded the cleaned text corpus (`rsc_clean.csv`) and
         - Engineered metadata features such as document year and document length.
         - Loaded three structured backend metadata tables that the chatbot will use for numeric / fact-style queries
           (Federal Register tracking and major rules).
         - Built TF–IDF feature representations and trained baseline rule-based classifiers (Multinomial Naive Bayes and
           Logistic Regression).
         - Defined a recurrent LSTM classifier and an autoencoder model
         - Generated transformer-based sentence embeddings for each RSC document using a pretrained MiniLM model and built
           a nearest-neighbor retrieval index to support the Retrieval-Augmented Generation (RAG) chatbot.
         - Saved all relevant models, vectorizers, embeddings, and manifests to disk so that the Streamlit app can reuse
           them without recomputing.

The overall goal of this module was to make RSC publications searchable, interpretable, and ready for interactive,
explainable retrieval in RegPolicyBot.
========================================================================================================================
"""
# ======================================================================================================================
# SECTION 0: Imports, Configuration, and Paths

# In this section, I imported all required libraries, configured environment settings, and established project paths.
# I also initialized runtime flags indicating which models or components would run based on available dependencies.
# This setup phase ensured that the modeling environment was correctly configured before any processing or training began.
# ======================================================================================================================

import os
import gc
import json
import nltk
import joblib
import warnings
import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from nltk.tokenize import sent_tokenize
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    confusion_matrix,
)

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import (
        Embedding,
        LSTM,
        Bidirectional,
        Dense,
        Dropout,
        Conv1D,
        MaxPooling1D,
        GlobalMaxPooling1D,
        GlobalAveragePooling1D,
        Input,
    )

    HAS_TF = True
    try:
        tf.config.optimizer.set_jit(True)
    except Exception:
        pass
except Exception:
    HAS_TF = False
    warnings.warn("TensorFlow was not available; deep learning models were not run.")

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except Exception:
    HAS_SENTENCE_TRANSFORMERS = False
    warnings.warn("sentence-transformers was not available; transformer embeddings were not run.")

try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False
    warnings.warn("SHAP was not available; SHAP explanations were not run.")

try:
    from lime.lime_text import LimeTextExplainer
    HAS_LIME = True
except Exception:
    HAS_LIME = False
    warnings.warn("LIME was not available; LIME explanations were not run.")

try:
    import torch
except Exception:
    torch = None

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except Exception:
    HAS_PLOTTING = False
    warnings.warn("matplotlib/seaborn not available; confusion matrix plot will be skipped.")

from datasets import Dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForMaskedLM,
    DistilBertForSequenceClassification,
    TrainingArguments,
    Trainer,
)

from scipy.special import softmax

RUN_LSTM = True and HAS_TF                                                      # Runtime flags
RUN_DENSE = True and HAS_TF
RUN_CNN = True and HAS_TF
RUN_BILSTM = True and HAS_TF
RUN_AUTOENCODER = True and HAS_TF
RUN_SHAP = True and HAS_SHAP
RUN_LIME = True and HAS_LIME
RUN_TRANSFORMER_EMBEDDINGS = True and HAS_SENTENCE_TRANSFORMERS
RUN_TRANSFORMER_FINETUNE = True and HAS_SENTENCE_TRANSFORMERS
RUN_DAPT = False and HAS_SENTENCE_TRANSFORMERS
RANDOM_STATE = 42
TEST_SIZE = 0.2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(BASE_DIR)
CODE_DIR = os.path.join(BASE_DIR, "code")
DATA_RAW_DIR = os.path.join(BASE_DIR, "data_raw")
DATA_CLEAN_DIR = os.path.join(BASE_DIR, "data_clean")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
BACKEND_META_DIR = os.path.join(BASE_DIR, "backend_metadata")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
CLEAN_DATA_PATH = os.path.join(DATA_CLEAN_DIR, "rsc_clean.csv")

# ======================================================================================================================
# SECTION 1: Load Cleaned Corpus and Engineer Metadata

# In this section, I loaded the cleaned RSC corpus produced in the preprocessing stage. I engineered additional metadata
# such as token-based document length and publication year. I also applied label encoding to category labels, producing
# numerical identifiers required for downstream classifiers. This step prepared the dataset for chunking, vectorization,
# and training.
# ======================================================================================================================

print("[INFO] Loading cleaned RSC corpus...")
df = pd.read_csv(CLEAN_DATA_PATH)

df = df.reset_index(drop=True)
df["doc_id"] = df.index.astype(int)

print("[INFO] Engineering metadata features (year and length)...")
df["date_parsed"] = pd.to_datetime(df["date"], errors="coerce")
df["document_year"] = df["date_parsed"].dt.year.fillna(-1).astype(int)

df["clean_text"] = df["clean_text"].fillna("").astype(str)
df["category"] = df["category"].fillna("unknown").astype(str)

# Document length in tokens
df["document_length"] = df["clean_text"].apply(
    lambda x: len(x.split()) if isinstance(x, str) else 0
)

label_encoder = LabelEncoder()                                                       # Label encoding at DOCUMENT level
df["category_id"] = label_encoder.fit_transform(df["category"])
num_classes = len(label_encoder.classes_)

print("[INFO] Discovered categories:")
for idx, name in enumerate(label_encoder.classes_):
    count = (df["category_id"] == idx).sum()
    print(f"  - {idx}: {name} (n={count})")
# ======================================================================================================================
# SECTION 1B: Domain-Adaptive Pretraining (optional)

# This section performed optional Domain-Adaptive Pretraining (DAPT) on DistilBERT using the full RSC corpus. When enabled,
# the script tokenized the text, prepared masked-language-modeling batches, and trained a domain-adapted DistilBERT model.
# This adaptation aimed to improve downstream performance by aligning the model with regulatory language.
# ======================================================================================================================
if RUN_DAPT and HAS_SENTENCE_TRANSFORMERS:
    print("[INFO] Running domain-adaptive pretraining (DAPT) on all RSC texts...")
    rsc_corpus = df["clean_text"].tolist()
    dapt_ds = Dataset.from_dict({"text": rsc_corpus})
    dapt_tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    def dapt_tokenize(batch):
        return dapt_tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=256,
        )
    tokenized = dapt_ds.map(dapt_tokenize, batched=True)
    mlm_model = DistilBertForMaskedLM.from_pretrained("distilbert-base-uncased")
    from transformers import DataCollatorForLanguageModeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=dapt_tokenizer, mlm_probability=0.15
    )
    dapt_args = TrainingArguments(
        output_dir=os.path.join(MODELS_DIR, "dapt_distilbert"),
        per_device_train_batch_size=8,
        num_train_epochs=1,
        learning_rate=5e-5,
        save_total_limit=1,
        logging_steps=100,
    )
    dapt_trainer = Trainer(
        model=mlm_model,
        args=dapt_args,
        train_dataset=tokenized,
        data_collator=data_collator,
    )

    dapt_trainer.train()
    mlm_model.save_pretrained(os.path.join(MODELS_DIR, "dapt_distilbert"))
    dapt_tokenizer.save_pretrained(os.path.join(MODELS_DIR, "dapt_distilbert"))
    print("[INFO] DAPT complete. Saved adapted DistilBERT.")
else:
    print("[INFO] Skipped DAPT (RUN_DAPT=False or sentence-transformers unavailable).")
# ======================================================================================================================
# SECTION 1C: TEXT CHUNKING (used by ALL classifiers and transformer fine-tuning)

# Here, I split each article into sentence-based chunks to address length constraints associated with NLP models. The
# chunking process converted long documents into smaller units of approximately equal size, enabling better classifier
# performance and preventing transformer truncation. Chunk-level class weights were computed afterward to address
# imbalance.
# ======================================================================================================================
print("[INFO] Creating text chunks for improved classification...")
nltk.download("punkt")

def chunk_text(text, max_tokens=250):
    sentences = sent_tokenize(text)
    current_chunk = []
    current_len = 0
    for sent in sentences:
        tokens = sent.split()
        if current_len + len(tokens) <= max_tokens:
            current_chunk.append(sent)
            current_len += len(tokens)
        else:
            if current_chunk:
                yield " ".join(current_chunk)
            current_chunk = [sent]
            current_len = len(tokens)

    if current_chunk:
        yield " ".join(current_chunk)
expanded_rows = []
for idx, row in df.iterrows():
    chunks = list(chunk_text(row["clean_text"], max_tokens=250))
    if not chunks:

        chunks = [row["clean_text"]]                                              # Fallback to use full text if chunking produced nothing
    for chunk in chunks:
        expanded_rows.append(
            {
                "text_chunk": chunk,
                "orig_doc_id": row["doc_id"],
                "orig_category": row["category"],
                "category_id": row["category_id"],
            }
        )
df_chunks = pd.DataFrame(expanded_rows)

df_chunks["text_hash"] = df_chunks["text_chunk"].apply(lambda x: hash(x))              # Removing duplicate chunks by content hash
df_chunks = df_chunks.drop_duplicates(subset="text_hash").drop(columns=["text_hash"])
df_chunks = df_chunks.reset_index(drop=True)
print(f"[INFO] Chunking complete: {len(df_chunks)} chunks created from {len(df)} documents.")
print("[INFO] Example chunk rows:", df_chunks.head(3).to_dict(orient="records"))      # Class weights based on chunk distribution
y_all_chunks = df_chunks["category_id"].values
unique_classes, counts = np.unique(y_all_chunks, return_counts=True)

print("\n[INFO] Chunk-level class distribution:")
for idx, cnt in zip(unique_classes, counts):
    print(f"  - {label_encoder.classes_[idx]}: {cnt} chunks")
class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=unique_classes,
    y=y_all_chunks,
)
CLASS_WEIGHTS = {int(cls): float(w) for cls, w in zip(unique_classes, class_weights_array)}
print("[INFO] Computed class weights:", CLASS_WEIGHTS)
# ======================================================================================================================
# SECTION 1D: Backend Metadata Tables (for numeric queries, not modeling)

# In this section, I loaded auxiliary backend metadata tables containing structured regulatory information such as datasets
# on major rules, Federal Register trends, and rulemaking counts. These tables were not used for text classification but
# were stored for numeric queries in the chatbot. A manifest summarizing table structure was also generated.
# ======================================================================================================================

backend_metadata = {}

def load_backend_csv(name: str, filename: str):
    path = os.path.join(BACKEND_META_DIR, filename)
    if os.path.exists(path):
        backend_metadata[name] = pd.read_csv(
            path, encoding="latin1", encoding_errors="replace"
        )
        print(f"[INFO] Loaded backend metadata '{name}' with shape {backend_metadata[name].shape}.")
    else:
        print(f"[WARN] Backend metadata file not found: {path}")

load_backend_csv("federal_register_tracking", "fr_tracking.csv")
load_backend_csv("major_rules_by_year", "major_rules_by_presidential_year.csv")
load_backend_csv("federal_register_rules_by_year", "federal_register_rules_by_presidential_year.csv")

backend_manifest = {
    "tables": {
        name: {"rows": int(df_meta.shape[0]), "cols": list(df_meta.columns)}
        for name, df_meta in backend_metadata.items()
    }
}
with open(
    os.path.join(RESULTS_DIR, "backend_metadata_manifest.json"),
    "w",
    encoding="utf-8",
) as f:
    json.dump(backend_manifest, f, indent=2)
print("[INFO] Saved backend metadata manifest.")

# ======================================================================================================================
# SECTION 2: Train/Test Split (CHUNKS) and TF–IDF Features

# This section created the train/test split at the chunk level using stratified sampling to preserve category distribution.
# I generated TF–IDF feature vectors and applied RandomOverSampler to balance classes in the training set. These features
# served as inputs for baseline classifiers and ensured fair evaluation despite dataset imbalance.
# ======================================================================================================================
indices = np.arange(len(df_chunks))
train_idx, test_idx = train_test_split(
    indices,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=df_chunks["category_id"],
)
X_train_text = df_chunks.loc[train_idx, "text_chunk"].values
X_test_text = df_chunks.loc[test_idx, "text_chunk"].values
y_train = df_chunks.loc[train_idx, "category_id"].values
y_test = df_chunks.loc[test_idx, "category_id"].values

print("[INFO] Fitting TF–IDF vectorizer on training chunks...")
tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.85,
    stop_words="english",
    sublinear_tf=True,
)
X_train_tfidf = tfidf.fit_transform(X_train_text)
X_test_tfidf = tfidf.transform(X_test_text)
ros = RandomOverSampler(random_state=RANDOM_STATE)                                     # Oversampling to handle class imbalance (chunk-level)
X_train_bal, y_train_bal = ros.fit_resample(X_train_tfidf, y_train)

print("[INFO] Oversampling complete:")
print(f"  Original training size: {X_train_tfidf.shape}")
print(f"  Balanced training size: {X_train_bal.shape}")
print(
    f"[INFO] TF–IDF train shape: {X_train_tfidf.shape}, test shape: {X_test_tfidf.shape}"
)
# ======================================================================================================================
# SECTION 3: Baseline Rule-Based Models (NB, LR, SVC, Voting)

# In this section, I trained and evaluated several baseline text classifiers including Multinomial Naive Bayes, Logistic
# Regression, and Linear SVC. I also created a soft-voting ensemble combining Naive Bayes and Logistic Regression. These
# baselines established reference performance levels prior to testing deep learning and transformer-based methods.
# ======================================================================================================================
baseline_results = {}
print("[INFO] Training Multinomial Naive Bayes baseline...")
nb_clf = MultinomialNB()
nb_clf.fit(X_train_bal, y_train_bal)
y_pred_nb = nb_clf.predict(X_test_tfidf)
f1_nb = f1_score(y_test, y_pred_nb, average="macro")
baseline_results["naive_bayes_f1_macro"] = float(f1_nb)

print("[RESULT] Naive Bayes F1-macro:", f1_nb)
print(
    classification_report(
        y_test, y_pred_nb, target_names=label_encoder.classes_
    )
)
joblib.dump(nb_clf, os.path.join(MODELS_DIR, "nb_tfidf_model.joblib"))

print("[INFO] Training Logistic Regression baseline...")
logreg_clf = LogisticRegression(max_iter=500, n_jobs=-1)
logreg_clf.fit(X_train_bal, y_train_bal)
y_pred_lr = logreg_clf.predict(X_test_tfidf)
f1_lr = f1_score(y_test, y_pred_lr, average="macro")
baseline_results["logreg_f1_macro"] = float(f1_lr)
print("[RESULT] Logistic Regression F1-macro:", f1_lr)
print(
    classification_report(
        y_test, y_pred_lr, target_names=label_encoder.classes_
    )
)
joblib.dump(logreg_clf, os.path.join(MODELS_DIR, "logreg_tfidf_model.joblib"))
joblib.dump(tfidf, os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib"))
joblib.dump(label_encoder, os.path.join(MODELS_DIR, "label_encoder.joblib"))

print("[INFO] Training Linear SVC baseline...")
svc_clf = LinearSVC()
svc_clf.fit(X_train_bal, y_train_bal)
y_pred_svc = svc_clf.predict(X_test_tfidf)
f1_svc = f1_score(y_test, y_pred_svc, average="macro")
baseline_results["svc_f1_macro"] = float(f1_svc)

print("[RESULT] LinearSVC F1-macro:", f1_svc)
print(
    classification_report(
        y_test, y_pred_svc, target_names=label_encoder.classes_
    )
)
joblib.dump(svc_clf, os.path.join(MODELS_DIR, "svc_tfidf_model.joblib"))

print("[INFO] Training soft Voting ensemble (NB + LR)...")
voting_clf = VotingClassifier(
    estimators=[("nb", nb_clf), ("lr", logreg_clf)],
    voting="soft",
    n_jobs=-1,
)
voting_clf.fit(X_train_bal, y_train_bal)
y_pred_vote = voting_clf.predict(X_test_tfidf)
f1_vote = f1_score(y_test, y_pred_vote, average="macro")
baseline_results["voting_soft_f1_macro"] = float(f1_vote)

print("[RESULT] Voting (soft) F1-macro:", f1_vote)
print(
    classification_report(
        y_test, y_pred_vote, target_names=label_encoder.classes_
    )
)
joblib.dump(
    voting_clf,
    os.path.join(MODELS_DIR, "voting_nb_logreg_tfidf_model.joblib"),
)
# ======================================================================================================================
# SECTION 3X: Deep Learning Models (Dense, CNN-1D, BiLSTM)

# This section trained deep learning models on padded token sequences using TensorFlow/Keras. Models included a Dense
# Neural Network, a 1D Convolutional Neural Network, and a Bidirectional LSTM. These architectures were evaluated on F1
# macro scores to compare their effectiveness against TF–IDF baselines and transformer models.
# ======================================================================================================================
deep_scores = {}
if HAS_TF:
    print("[INFO] Preparing shared tokenizer for deep learning models...")
    MAX_WORDS = 30000
    MAX_LEN = 400
    tokenizer_dl = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tokenizer_dl.fit_on_texts(X_train_text)
    X_train_seq_dl = tokenizer_dl.texts_to_sequences(X_train_text)
    X_test_seq_dl = tokenizer_dl.texts_to_sequences(X_test_text)
    X_train_pad_dl = pad_sequences(
        X_train_seq_dl, maxlen=MAX_LEN, padding="post", truncating="post"
    )
    X_test_pad_dl = pad_sequences(
        X_test_seq_dl, maxlen=MAX_LEN, padding="post", truncating="post"
    )

    # ---------- Dense Neural Network ---------------------------------------
    if RUN_DENSE:
        print("[INFO] Training Dense Neural Network...")
        dense_model = Sequential(
            [
                Embedding(MAX_WORDS, 128, input_length=MAX_LEN),
                GlobalAveragePooling1D(),
                Dense(128, activation="relu"),
                Dropout(0.3),
                Dense(num_classes, activation="softmax"),
            ]
        )
        dense_model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer="adam",
            metrics=["accuracy"],
        )
        dense_model.fit(
            X_train_pad_dl,
            y_train,
            epochs=3,
            batch_size=32,
            validation_split=0.1,
            class_weight=CLASS_WEIGHTS,
            verbose=1,
        )
        y_pred_dense = dense_model.predict(X_test_pad_dl).argmax(axis=1)
        f1_dense = f1_score(y_test, y_pred_dense, average="macro")
        deep_scores["dense_f1_macro"] = float(f1_dense)
        print("[RESULT] Dense Model F1-macro:", f1_dense)

    # ---------- CNN-1D -----------------------------------------------                 # Just Experimenting though its an image model
    if RUN_CNN:
        print("[INFO] Training CNN-1D model...")
        cnn_model = Sequential(
            [
                Embedding(MAX_WORDS, 128, input_length=MAX_LEN),
                Conv1D(128, 5, activation="relu"),
                MaxPooling1D(pool_size=2),
                GlobalMaxPooling1D(),
                Dense(128, activation="relu"),
                Dropout(0.3),
                Dense(num_classes, activation="softmax"),
            ]
        )
        cnn_model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer="adam",
            metrics=["accuracy"],
        )
        cnn_model.fit(
            X_train_pad_dl,
            y_train,
            epochs=3,
            batch_size=32,
            validation_split=0.1,
            class_weight=CLASS_WEIGHTS,
            verbose=1,
        )
        y_pred_cnn = cnn_model.predict(X_test_pad_dl).argmax(axis=1)
        f1_cnn = f1_score(y_test, y_pred_cnn, average="macro")
        deep_scores["cnn_f1_macro"] = float(f1_cnn)
        print("[RESULT] CNN-1D F1-macro:", f1_cnn)

    # ---------- BiLSTM --------------------------------------------------------------
    if RUN_BILSTM:
        print("[INFO] Training BiLSTM model...")
        bilstm_model = Sequential(
            [
                Embedding(MAX_WORDS, 128, input_length=MAX_LEN),
                Bidirectional(LSTM(128, return_sequences=False)),
                Dropout(0.3),
                Dense(128, activation="relu"),
                Dropout(0.3),
                Dense(num_classes, activation="softmax"),
            ]
        )
        bilstm_model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer="adam",
            metrics=["accuracy"],
        )
        bilstm_model.fit(
            X_train_pad_dl,
            y_train,
            epochs=3,
            batch_size=32,
            validation_split=0.1,
            class_weight=CLASS_WEIGHTS,
            verbose=1,
        )
        y_pred_bilstm = bilstm_model.predict(X_test_pad_dl).argmax(axis=1)
        f1_bilstm = f1_score(y_test, y_pred_bilstm, average="macro")
        deep_scores["bilstm_f1_macro"] = float(f1_bilstm)
        print("[RESULT] BiLSTM F1-macro:", f1_bilstm)
else:
    print("[INFO] Skipped Dense/CNN/BiLSTM (TensorFlow unavailable).")
# ======================================================================================================================
# SECTION 4: LSTM Classifier (Recurrent Network)

# Here, I trained a standalone LSTM classifier to capture long-range dependencies in text sequences. After preparing
# padded inputs and fitting the model on category labels, I evaluated its performance and saved the trained model and
# tokenizer for later inference.
# ======================================================================================================================
if RUN_LSTM and HAS_TF:
    print("[INFO] Training standalone LSTM classifier...")
    X_train_pad = X_train_pad_dl
    X_test_pad = X_test_pad_dl
    lstm_model = Sequential(
        [
            Embedding(input_dim=MAX_WORDS, output_dim=128, input_length=MAX_LEN),
            LSTM(128, return_sequences=False),
            Dropout(0.3),
            Dense(64, activation="relu"),
            Dropout(0.3),
            Dense(num_classes, activation="softmax"),
        ]
    )
    lstm_model.compile(
        loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    lstm_model.fit(
        X_train_pad,
        y_train,
        validation_split=0.1,
        epochs=3,
        batch_size=32,
        class_weight=CLASS_WEIGHTS,
        verbose=1,
    )

    y_pred_lstm = lstm_model.predict(X_test_pad).argmax(axis=1)
    f1_lstm = f1_score(y_test, y_pred_lstm, average="macro")
    deep_scores["lstm_f1_macro"] = float(f1_lstm)
    print("[RESULT] LSTM F1-macro:", f1_lstm)
    print(
        classification_report(
            y_test, y_pred_lstm, target_names=label_encoder.classes_
        )
    )
    lstm_model.save(os.path.join(MODELS_DIR, "lstm_text_classifier.keras"))
    joblib.dump(
        tokenizer_dl, os.path.join(MODELS_DIR, "lstm_tokenizer.joblib")
    )
else:
    print("[INFO] Skipped LSTM training (RUN_LSTM=False or TensorFlow unavailable).")
# ======================================================================================================================
# SECTION 5: Autoencoder for TF–IDF Latent Topics

# This section trained an autoencoder on TF–IDF vectors to learn compressed latent representations of document chunks.
# The autoencoder provided an unsupervised perspective on topic structure and could be used for visualization or
# clustering. Both the compressed latent vectors and the autoencoder model were saved.
# ======================================================================================================================
if RUN_AUTOENCODER and HAS_TF:
    print("[INFO] Training autoencoder on TF–IDF chunk features...")
    X_train_dense = X_train_tfidf.toarray()
    input_dim = X_train_dense.shape[1]
    latent_dim = 64

    input_layer = Input(shape=(input_dim,))
    encoded = Dense(256, activation="relu")(input_layer)
    encoded = Dense(latent_dim, activation="relu")(encoded)
    decoded = Dense(256, activation="relu")(encoded)
    decoded = Dense(input_dim, activation="sigmoid")(decoded)
    autoencoder = Model(inputs=input_layer, outputs=decoded)
    autoencoder.compile(optimizer="adam", loss="binary_crossentropy")
    autoencoder.fit(
        X_train_dense,
        X_train_dense,
        epochs=5,
        batch_size=64,
        shuffle=True,
        validation_split=0.1,
        verbose=1,
    )
    encoder = Model(inputs=input_layer, outputs=encoded)
    latent_train = encoder.predict(X_train_dense)
    np.save(
        os.path.join(MODELS_DIR, "tfidf_autoencoder_latent.npy"), latent_train
    )
    autoencoder.save(os.path.join(MODELS_DIR, "tfidf_autoencoder.keras"))
    print("[INFO] Saved autoencoder and latent representation.")
else:
    print(
        "[INFO] Skipped autoencoder training (RUN_AUTOENCODER=False or TensorFlow unavailable)."
    )
# ======================================================================================================================
# SECTION 5B: Clear GPU memory before BERT fine-tuning (best-effort, safe)

# This section attempted to free GPU memory prior to transformer fine-tuning. Although not always necessary, clearing
# the CUDA cache helped prevent memory fragmentation and reduced the likelihood of out-of-memory errors during DistilBERT
# training.
# ======================================================================================================================
print("[INFO] Clearing GPU memory before DistilBERT fine-tuning (best-effort)...")
try:
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
except Exception as e:
    print(f"[WARN] GPU clearing (torch) skipped due to: {e}")
# ======================================================================================================================
# SECTION 5C: Transformer Fine-Tuning (DistilBERT) on CHUNKS

# In this section, I fine-tuned a DistilBERT classification model on chunk-level text. The script prepared Hugging Face
# datasets, tokenized inputs, initialized training arguments, and trained the classifier using the Trainer API. Evaluation
# metrics, a confusion matrix, and model weights were saved for downstream use.
# ======================================================================================================================
bert_probs = None                                                                       # Refilling after prediction
if RUN_TRANSFORMER_FINETUNE and HAS_SENTENCE_TRANSFORMERS:
    print("[INFO] Fine-tuning DistilBERT classification model on chunks...")
    dapt_path = os.path.join(MODELS_DIR, "dapt_distilbert")                             # Using DAPT model if available, otherwise base DistilBERT
    base_model_name = dapt_path if os.path.exists(dapt_path) else "distilbert-base-uncased"
    tokenizer_bert = DistilBertTokenizerFast.from_pretrained(base_model_name)
    bert_model = DistilBertForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=num_classes,
    )

    train_texts = df_chunks.loc[train_idx, "text_chunk"].tolist()                        # Building train/test datasets from CHUNK SPLIT
    test_texts = df_chunks.loc[test_idx, "text_chunk"].tolist()
    train_labels = df_chunks.loc[train_idx, "category_id"].astype(int).tolist()
    test_labels = df_chunks.loc[test_idx, "category_id"].astype(int).tolist()
    train_ds = Dataset.from_dict({"text": train_texts, "labels": train_labels})
    test_ds = Dataset.from_dict({"text": test_texts, "labels": test_labels})
    def encode_batch(batch):
        return tokenizer_bert(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=256,
        )

    train_ds = train_ds.map(encode_batch, batched=True)
    test_ds = test_ds.map(encode_batch, batched=True)
    train_ds = train_ds.remove_columns(["text"])
    test_ds = test_ds.remove_columns(["text"])
    train_ds.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels"]
    )
    test_ds.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels"]
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        f1 = f1_score(labels, preds, average="macro")
        acc = accuracy_score(labels, preds)
        prec = precision_score(labels, preds, average="macro")
        rec = recall_score(labels, preds, average="macro")
        return {
            "accuracy": acc,
            "f1_macro": f1,
            "precision_macro": prec,
            "recall_macro": rec,
        }

    bert_args = TrainingArguments(
        output_dir=os.path.join(MODELS_DIR, "distilbert_classifier"),
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_steps=100,
        save_total_limit=1,
        do_eval=True,
    )

    bert_trainer = Trainer(
        model=bert_model,
        args=bert_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=compute_metrics,
    )
    bert_trainer.train()
    bert_eval = bert_trainer.evaluate()
    print("[RESULT] DistilBERT eval metrics:", bert_eval)
    predictions = bert_trainer.predict(test_ds)
    pred_labels = predictions.predictions.argmax(axis=-1)
    true_labels = np.array(test_labels)
    bert_f1 = f1_score(true_labels, pred_labels, average="macro")
    bert_precision = precision_score(true_labels, pred_labels, average="macro")
    bert_recall = recall_score(true_labels, pred_labels, average="macro")
    bert_accuracy = accuracy_score(true_labels, pred_labels)
    print("\n[RESULT] DistilBERT classification report:")
    print(
        classification_report(
            true_labels, pred_labels, target_names=label_encoder.classes_
        )
    )

    if HAS_PLOTTING:
        cm = confusion_matrix(true_labels, pred_labels)
        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_,
        )
        plt.title("DistilBERT Confusion Matrix (Chunks)")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        cm_path = os.path.join(RESULTS_DIR, "distilbert_confusion_matrix.png")
        plt.savefig(cm_path)
        plt.close()
        print(f"[INFO] Saved confusion matrix: {cm_path}")

    bert_logits = predictions.predictions                                                # probabilities for potential ensemble
    bert_probs = softmax(bert_logits, axis=1)
    baseline_results["bert_f1_macro"] = float(bert_f1)
    baseline_results["bert_precision_macro"] = float(bert_precision)
    baseline_results["bert_recall_macro"] = float(bert_recall)
    baseline_results["bert_accuracy"] = float(bert_accuracy)
    bert_model.save_pretrained(os.path.join(MODELS_DIR, "bert_finetuned"))               # Saving model + tokenizer
    tokenizer_bert.save_pretrained(os.path.join(MODELS_DIR, "bert_tokenizer"))
else:
    print(
        "[INFO] Skipped DistilBERT fine-tuning (RUN_TRANSFORMER_FINETUNE=False or sentence-transformers unavailable)."
    )
# ======================================================================================================================
# SECTION 6: Transformer Embeddings & Retrieval Index for RAG (MiniLM on FULL DOCUMENTS)

# This section generated MiniLM sentence embeddings for the full corpus to support retrieval-augmented generation (RAG).
# After computing embeddings, I trained a NearestNeighbors index for semantic retrieval, demonstrated example queries,
# and saved all retrieval artifacts for integration into the Streamlit chatbot.
# ======================================================================================================================
if RUN_TRANSFORMER_EMBEDDINGS and HAS_SENTENCE_TRANSFORMERS:
    print("[INFO] Generating MiniLM embeddings for full corpus (RAG backend)...")
    sent_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    corpus_texts = df["clean_text"].tolist()
    embeddings = sent_model.encode(
        corpus_texts, batch_size=32, show_progress_bar=True
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)
    np.save(
        os.path.join(MODELS_DIR, "rsc_miniLM_embeddings.npy"), embeddings
    )

    print("[INFO] Building NearestNeighbors index for retrieval...")
    nn_index = NearestNeighbors(n_neighbors=10, metric="cosine", n_jobs=-1)
    nn_index.fit(embeddings)

    example_query = "What are the major regulatory trends?"                                  # Sample retrieval demo
    query_emb = sent_model.encode([example_query])
    distances, indices_knn = nn_index.kneighbors(query_emb, n_neighbors=5)
    retrieval_results = []
    for rank, (dist, idx) in enumerate(
        zip(distances[0], indices_knn[0]), start=1
    ):
        retrieval_results.append(
            {
                "rank": rank,
                "doc_index": int(idx),
                "cosine_similarity": float(1 - dist),
                "title": df.loc[idx, "title"],
                "category": df.loc[idx, "category"],
                "text_preview": df.loc[idx, "clean_text"][:200],
            }
        )

    with open(
        os.path.join(RESULTS_DIR, "retrieval_demo.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(retrieval_results, f, indent=2)
    print("[INFO] Saved retrieval demo with cosine similarity scores.")
    joblib.dump(
        nn_index,
        os.path.join(MODELS_DIR, "rsc_embeddings_nn_index.joblib"),
    )
    retrieval_manifest = df[
        [
            "doc_id",
            "category",
            "title",
            "author",
            "date",
            "url",
            "document_year",
            "document_length",
        ]
    ].copy()
    retrieval_manifest.to_csv(
        os.path.join(RESULTS_DIR, "rsc_retrieval_manifest.csv"),
        index=False,
    )
    print("[INFO] Saved transformer embeddings and retrieval manifest.")
else:
    print(
        "[INFO] Skipped transformer embeddings (RUN_TRANSFORMER_EMBEDDINGS=False or sentence-transformers unavailable)."
    )
# ======================================================================================================================
# SECTION 6B: SBERT Embeddings for Classification (Chunks)

# In this section, I computed SHAP values for the Logistic Regression classifier to interpret which TF–IDF features most
# strongly influenced predictions. I saved feature importance summaries for inclusion in reports and visual dashboards.
# ======================================================================================================================
if HAS_SENTENCE_TRANSFORMERS:
    print("[INFO] Building SBERT embeddings for CHUNK classification...")
    sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
    X_sbert = sbert_model.encode(
        df_chunks["text_chunk"].tolist(), show_progress_bar=True
    )
    X_sbert = np.asarray(X_sbert, dtype=np.float32)
    X_train_sbert = X_sbert[train_idx]
    X_test_sbert = X_sbert[test_idx]
    y_train_sbert = y_train
    y_test_sbert = y_test
    sbert_clf = LogisticRegression(max_iter=3000)
    sbert_clf.fit(X_train_sbert, y_train_sbert)
    sbert_pred = sbert_clf.predict(X_test_sbert)
    sbert_f1 = f1_score(y_test_sbert, sbert_pred, average="macro")

    print("[RESULT] SBERT classifier F1-macro:", sbert_f1)
    baseline_results["sbert_f1_macro"] = float(sbert_f1)

else:
    print("[INFO] Skipped SBERT classifier (sentence-transformers unavailable).")
# ======================================================================================================================
# SECTION 7: SHAP Explainability for Logistic Regression
# ======================================================================================================================
if RUN_SHAP and HAS_SHAP:
    print("[INFO] Running SHAP explainability for Logistic Regression...")
    try:
        n_background = min(200, X_train_bal.shape[0])
        n_explain = min(200, X_test_tfidf.shape[0])
        rng = np.random.default_rng(RANDOM_STATE)
        background_idx = rng.choice(
            X_train_bal.shape[0], size=n_background, replace=False
        )
        explain_idx = rng.choice(
            X_test_tfidf.shape[0], size=n_explain, replace=False
        )

        background_data = X_train_bal[background_idx]
        explain_data = X_test_tfidf[explain_idx]
        explainer = shap.LinearExplainer(logreg_clf, background_data)
        shap_values = explainer.shap_values(explain_data)
        np.save(
            os.path.join(RESULTS_DIR, "shap_values_logreg.npy"),
            shap_values,
        )
        feature_names = np.array(tfidf.get_feature_names_out())

        if isinstance(shap_values, list):
            mean_abs = np.mean(
                [np.mean(np.abs(sv), axis=0) for sv in shap_values], axis=0
            )
        else:
            mean_abs = np.mean(np.abs(shap_values), axis=0)
        top_k = 30
        top_idx = np.argsort(mean_abs)[-top_k:][::-1]
        shap_summary = [
            {"feature": feature_names[i], "mean_abs_shap": float(mean_abs[i])}
            for i in top_idx
        ]
        with open(
            os.path.join(RESULTS_DIR, "shap_top_features_logreg.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(shap_summary, f, indent=2)

        print("[INFO] SHAP values computed and saved.")
    except Exception as e:
        print(f"[WARN] SHAP analysis skipped due to error: {e}")
else:
    print("[INFO] Skipped SHAP explanations (RUN_SHAP=False or SHAP unavailable).")
# ======================================================================================================================
# SECTION 8: LIME Text Explanation for One Example

# This section generated a LIME explanation for a single test example, showing which words most affected the classifier’s
# output. The explanation was exported as an HTML visualization to support interpretability in the Streamlit interface.
# ======================================================================================================================
if RUN_LIME and HAS_LIME:
    print("[INFO] Running LIME text explanation for one example...")
    try:
        class_names = list(label_encoder.classes_)
        def predict_proba_for_lime(texts):
            X_tfidf = tfidf.transform(texts)
            return logreg_clf.predict_proba(X_tfidf)
        explainer = LimeTextExplainer(class_names=class_names)
        example_index = 0
        example_text = X_test_text[example_index]
        explanation = explainer.explain_instance(
            example_text,
            predict_proba_for_lime,
            num_features=20,
        )
        lime_html_path = os.path.join(
            RESULTS_DIR, "lime_example_logreg.html"
        )
        explanation.save_to_file(lime_html_path)
        print(f"[INFO] Saved LIME explanation to {lime_html_path}")
    except Exception as e:
        print(f"[WARN] LIME explanation skipped due to error: {e}")
else:
    print("[INFO] Skipped LIME explanations (RUN_LIME=False or LIME unavailable).")
# ======================================================================================================================
# SECTION 9: Combined Results JSON + Model Comparison Table

# In this section, I compiled all model metrics into a unified JSON file and produced a Markdown comparison table. These
# artifacts created a consolidated snapshot of baseline, deep learning, and transformer model performance.
# ======================================================================================================================
print("[INFO] Preparing combined results summary...")
combined_results = {
    "baselines": baseline_results,
    "deep_learning": deep_scores,
    "autoencoder": {"latent_dim": 64, "trained": RUN_AUTOENCODER and HAS_TF},
}
combined_json_path = os.path.join(RESULTS_DIR, "combined_model_results.json")
with open(combined_json_path, "w", encoding="utf-8") as f:
    json.dump(combined_results, f, indent=2)
print(f"[INFO] Saved: {combined_json_path}")
baseline_metrics_path = os.path.join(                                                        # Also saving baseline metrics alone (for quick reference)
    RESULTS_DIR, "baseline_classification_metrics.json"
)
with open(baseline_metrics_path, "w", encoding="utf-8") as f:
    json.dump(baseline_results, f, indent=2)
print(f"[INFO] Saved baseline metrics: {baseline_metrics_path}")
                                                                                             # Building a comparison table in Markdown
comparison_table = """                                                                      
Model | F1-macro | Notes
------|----------|-------------------------
Naive Bayes | {:.4f} | TF–IDF on chunks
Logistic Regression | {:.4f} | TF–IDF baseline
LinearSVC | {:.4f} | TF–IDF baseline
Voting Ensemble | {:.4f} | NB + LR (TF–IDF)
Dense Neural Network | {:.4f} | Deep model (chunks)
CNN-1D | {:.4f} | Deep model (chunks)
BiLSTM | {:.4f} | Deep model (chunks)
LSTM | {:.4f} | Recurrent model (chunks)
DistilBERT Fine-Tuned | {:.4f} | Transformer on chunks
SBERT Logistic | {:.4f} | SBERT embeddings + LR (chunks)
Autoencoder | - | Latent=64 (unsupervised)
""".format(
    baseline_results.get("naive_bayes_f1_macro", 0.0),
    baseline_results.get("logreg_f1_macro", 0.0),
    baseline_results.get("svc_f1_macro", 0.0),
    baseline_results.get("voting_soft_f1_macro", 0.0),
    deep_scores.get("dense_f1_macro", 0.0),
    deep_scores.get("cnn_f1_macro", 0.0),
    deep_scores.get("bilstm_f1_macro", 0.0),
    deep_scores.get("lstm_f1_macro", 0.0),
    baseline_results.get("bert_f1_macro", 0.0),
    baseline_results.get("sbert_f1_macro", 0.0),
)

table_path = os.path.join(RESULTS_DIR, "model_comparison_table.md")
with open(table_path, "w", encoding="utf-8") as f:
    f.write(comparison_table)
print(f"[INFO] Saved model comparison table: {table_path}")
# ======================================================================================================================
# SECTION 10: FINAL SUMMARY

# This final summary section printed key modeling statistics including the number of documents, number of chunks, model
# performance scores, and file paths of saved artifacts. This completed the modeling pipeline before moving on to the
# application interface.
# ======================================================================================================================
print("\n[SUMMARY] 4_modeling.py completed.")
print(f"  Documents processed:       {df.shape[0]}")
print(f"  Chunks processed:          {df_chunks.shape[0]}")
print(f"  Categories discovered:     {len(label_encoder.classes_)}")
print("  Baseline & transformer F1-macro scores:")
for k, v in baseline_results.items():
    if isinstance(v, dict):
        print(f"    - {k}: {v}")
    else:
        print(f"    - {k}: {v:.4f}")
print("  Deep learning F1-macro scores:")
for k, v in deep_scores.items():
    print(f"    - {k}: {v:.4f}")
print("  Models and artifacts were saved to:")
print(f"    MODELS_DIR  = {MODELS_DIR}")
print(f"    RESULTS_DIR = {RESULTS_DIR}")
print("\nNEXT: Use 5_streamlit_app.py to build the RegPolicyBot UI backed by these models and embeddings.")

# ======================================================================================================================
# SECTION 12 — CONCLUSIONS OF MODELING PIPELINE
# ======================================================================================================================
"""
This script completed the full modeling and embedding pipeline for the RegPolicyBot project. The workflow accomplished:

    • Loading and preparing cleaned regulatory text with engineered metadata.
    • Chunking all documents into manageable segments for classification and transformer processing.
    • Constructing TF–IDF representations and training baseline machine learning models.
    • Training multiple deep learning architectures including Dense, CNN-1D, BiLSTM, and LSTM networks.
    • Fine-tuning DistilBERT on chunk-level labels to build a high-performance transformer classifier.
    • Generating MiniLM sentence embeddings and building a semantic retrieval index for RAG.
    • Computing SHAP and LIME explanations for interpretability.
    • Saving all models, encoders, embeddings, and comparison metrics for downstream application use.

This modeling module prepared every artifact required by the Streamlit chatbot, enabling both predictive classification
and retrieval-augmented generation (RAG) capabilities. The module was completed successfully.
"""
# ======================================================================================================================
# SECTION 12 — CODE REFERENCES
# ======================================================================================================================
# These references document the external sources that informed the design, modeling strategies, transformer usage,
# tokenizer behavior, chunking decisions, classifier selection, and retrieval index construction in this script.
# ======================================================================================================================
# A. LOADING DATA & FEATURE ENGINEERING

# [1] Pandas Documentation – Working with Datetime and Metadata Features
#     https://pandas.pydata.org/docs/
# [2] Scikit-learn Documentation – Label Encoding and Feature Engineering
#     https://scikit-learn.org/stable/modules/preprocessing.html

# B. TEXT CHUNKING FOR NLP MODELS

# [1] NLTK Tokenizers – Sentence Tokenization (punkt)
#     https://www.nltk.org/api/nltk.tokenize.html
# [2] Research: Long-Document NLP Techniques & Chunking Strategies (ACL Anthology)
#     https://aclanthology.org/2021.acl-long.36/

# C. BACKEND METADATA LOADING

# [1] Data Engineering Best Practices for Auxiliary Metadata Integration (Google Cloud)
#     https://cloud.google.com/architecture
# [2] ACM Recommendations – Metadata for Reproducible NLP Systems
#     https://www.acm.org/publications/policies/artifact-review/

# D. TF-IDF, TRAIN/TEST SPLIT, OVERSAMPLING

# [1] Scikit-learn Documentation – TF-IDF Vectorizer
#     https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
# [2] Imbalanced-Learn Documentation – RandomOverSampler
#     https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.RandomOverSampler.html
# [3] Scikit-learn Documentation – Stratified Train/Test Split
#     https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html

# E. BASELINE MODELS (NB, LR, SVC, Voting)

# [1] Scikit-learn Documentation – Multinomial Naive Bayes, Logistic Regression, LinearSVC
#     https://scikit-learn.org/stable/supervised_learning.html
# [2] Scikit-learn Ensemble Methods – VotingClassifier
#     https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html
# [3] Research: TF-IDF + Linear Models as Baselines in Text Classification
#     https://aclanthology.org/P18-2018/

# F. DEEP LEARNING MODELS (Dense, CNN-1D, BiLSTM, LSTM)

# [1] TensorFlow Keras Documentation – Sequential API & Text Models
#     https://keras.io/guides/sequential_model/
# [2] Research: CNNs and LSTMs for Text Classification (Yoon Kim, 2014)
#     https://arxiv.org/abs/1408.5882
# [3] Research: Bidirectional LSTM for Document Classification
#     https://aclanthology.org/W16-1610/

# G. AUTOENCODER FOR TF-IDF LATENT TOPICS

# [1] TensorFlow Keras – Autoencoder Examples
#     https://keras.io/examples/keras_recipes/autoencoder/
# [2] Academic: Dimensionality Reduction with Autoencoders
#     https://www.cs.toronto.edu/~hinton/science.pdf

# H. DISTILBERT FINE-TUNING ON CHUNKS

# [1] Hugging Face Transformers Documentation – Trainer & Fine-Tuning
#     https://huggingface.co/docs/transformers/main_classes/trainer
# [2] DistilBERT Paper – Knowledge Distillation for Smaller Transformers
#     https://arxiv.org/abs/1910.01108
# [3] Research: Chunk-based Fine Tuning for Long-Document Classification
#     https://aclanthology.org/2020.acl-main.577/

# I. TRANSFORMER EMBEDDINGS (MiniLM) & RETRIEVAL INDEX

# [1] Sentence-Transformers Documentation – MiniLM Models
#     https://www.sbert.net/docs/pretrained_models.html
# [2] NearestNeighbors (Sklearn) – Cosine Similarity for RAG
#     https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html
# [3] Research: Dense Embedding Retrieval for Open-Domain QA
#     https://arxiv.org/abs/2004.04906

# J. SBERT + LOGISTIC REGRESSION CLASSIFIER

# [1] Sentence-Transformers – Embedding Generation Pipeline
#     https://www.sbert.net/
# [2] Research: Semantic Similarity Using SBERT
#     https://arxiv.org/abs/1908.10084

# K. SHAP EXPLAINABILITY

# [1] SHAP Documentation – LinearExplainer
#     https://shap.readthedocs.io/en/latest/
# [2] Academic: SHAP Values for Model Interpretability (Lundberg & Lee, 2017)
#     https://arxiv.org/abs/1705.07874

# L. LIME TEXT EXPLANATIONS

# [1] LIME Documentation – lime.lime_text
#     https://lime-ml.readthedocs.io/en/latest/lime.html#lime.lime_text.LimeTextExplainer
# [2] Original LIME Paper – Ribeiro, Singh, Guestrin (KDD 2016)
#     https://arxiv.org/abs/1602.04938

# M. MODEL RESULTS, METRICS, & F1 COMPUTATION

# [1] Scikit-learn Model Evaluation Metrics (F1, Precision, Recall)
#     https://scikit-learn.org/stable/modules/model_evaluation.html
# [2] Academic: Macro-F1 for Class-Imbalanced Text Classification
#     https://aclanthology.org/2020.lrec-1.577/

# N. RETRIEVAL-AUGMENTED GENERATION (RAG) BACKEND

# [1] Facebook AI – DPR / RAG Architecture Overview
#     https://arxiv.org/abs/2005.11401
# [2] Hugging Face – Retrieval-Augmented Generation Concepts
#     https://huggingface.co/docs/transformers/model_doc/rag
# ======================================================================================================================