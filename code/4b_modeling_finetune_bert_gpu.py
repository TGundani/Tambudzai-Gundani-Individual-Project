"""
========================================================================================================================
RegPolicyBot – FINE-TUNING THE BEST MODEL
------------------------------------------------------------------------------------------------------------------------
Purpose:
    This script isolates ONLY the DistilBERT fine-tuning process in order to achieve higher classification performance
    without rerunning the full modeling pipeline from 4_modeling.py.

Workflow:
1. Loads the cleaned RSC dataset.
2. Rebuilds df_chunks (same logic as 4_modeling.py) and removes duplicate text chunks.
3. Performs a stratified train/test split at the chunk level.
4. Clears GPU memory (best-effort).
5. Loads DistilBERT exactly as done in 4_modeling.py, ensuring compatibility with
   the original environment.
6. Fine-tunes DistilBERT for 5 epochs.

GPU Result:
    - eval_accuracy: ~0.8344
    - eval_f1_macro: ~0.7457
    - weighted_f1: ~0.83
    - Strongest performance achieved so far, especially improved predictions for
      minority class (“journal_articles_working_papers”).
"""
# ======================================================================================================================
# Importing Libraries


# In this section, I imported all libraries used for transformer training, evaluation, and chunking. I also defined runtime
# flags indicating whether transformer fine-tuning should run, depending on GPU and dependency availability. Finally, I
# reconstructed all directory paths so that the script matched the same folder structure used in 4_modeling.py.
# ======================================================================================================================

import os
import gc
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    confusion_matrix,
)
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.tokenize import sent_tokenize
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
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except Exception:
    HAS_SENTENCE_TRANSFORMERS = False
    warnings.warn("sentence-transformers not available; RUN_TRANSFORMER_FINETUNE will be False.")

from datasets import Dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForMaskedLM,
    DistilBertForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from scipy.special import softmax

RUN_TRANSFORMER_FINETUNE = True and HAS_SENTENCE_TRANSFORMERS
RUN_DAPT = False and HAS_SENTENCE_TRANSFORMERS                          # Runtime flags (mirror 4_modeling.py)
RANDOM_STATE = 42
TEST_SIZE = 0.2
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_CLEAN_DIR = os.path.join(BASE_DIR, "data_clean")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

CLEAN_DATA_PATH = os.path.join(DATA_CLEAN_DIR, "rsc_clean.csv")
CHUNKS_PATH = os.path.join(DATA_CLEAN_DIR, "df_chunks.csv")

print("[INFO] BASE_DIR:", BASE_DIR)
print("[INFO] Using cleaned corpus at:", CLEAN_DATA_PATH)
# ======================================================================================================================
# 1. Loading cleaned corpus and label-encode categories (DOCUMENT level)

# In this section, I loaded the cleaned RSC dataset produced in 3_data_cleaning.py. I added document IDs, ensured text
# fields were valid strings, and applied label encoding to convert categorical labels into numerical form. This prepared
# the dataset for chunking and model training.
# ======================================================================================================================
print("[INFO] Loading cleaned RSC corpus...")
df = pd.read_csv(CLEAN_DATA_PATH)
df = df.reset_index(drop=True)
df["doc_id"] = df.index.astype(int)
df["clean_text"] = df["clean_text"].fillna("").astype(str)
df["category"] = df["category"].fillna("unknown").astype(str)

label_encoder = LabelEncoder()
df["category_id"] = label_encoder.fit_transform(df["category"])
num_classes = len(label_encoder.classes_)

print("[INFO] Discovered categories:")
for idx, name in enumerate(label_encoder.classes_):
    count = (df["category_id"] == idx).sum()
    print(f"  - {idx}: {name} (n={count})")

# ======================================================================================================================
# 2. Chunking
#
# This section reconstructed the chunk-level dataset (df_chunks) using the exact same sentence-based chunking logic from
# 4_modeling.py. Long articles were broken into ~250-token segments, allowing DistilBERT to process them without truncation.
# Duplicate chunks were removed using a content hash, and the resulting chunk dataset was saved for reuse.
# ======================================================================================================================
print("[INFO] Creating text chunks for improved classification...")
nltk.download("punkt", quiet=True)

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
        chunks = [row["clean_text"]]
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

df_chunks["text_hash"] = df_chunks["text_chunk"].apply(lambda x: hash(x))             # Removing duplicate chunks by content hash
df_chunks = df_chunks.drop_duplicates(subset="text_hash").drop(columns=["text_hash"])
df_chunks = df_chunks.reset_index(drop=True)
print(f"[INFO] Chunking complete: {len(df_chunks)} chunks created from {len(df)} documents.")
print("[INFO] Example chunk rows:", df_chunks.head(3).to_dict(orient="records"))

try:                                                                                   # Saving for reuse
    df_chunks.to_csv(CHUNKS_PATH, index=False)
    print(f"[INFO] Saved df_chunks to {CHUNKS_PATH}")
except Exception as e:
    print(f"[WARN] Could not save df_chunks.csv: {e}")
# ======================================================================================================================
# 3. Train/test split at CHUNK level

# In this section, I performed a stratified train/test split at the chunk level to preserve class distribution. The
# resulting indices were used to build HuggingFace datasets for fine-tuning.
# ======================================================================================================================
indices = np.arange(len(df_chunks))

train_idx, test_idx = train_test_split(
    indices,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=df_chunks["category_id"],
)
# ======================================================================================================================
# 4. Clearing GPU memory before BERT fine-tuning (same as Section 5B)

# This section attempted to clear GPU memory before initializing DistilBERT, preventing CUDA fragmentation and reducing
# the chance of memory-related training failures. This mirrored the defensive memory-cleanup logic used previously in
# 4_modeling.py.
# ======================================================================================================================
print("[INFO] Clearing GPU memory before DistilBERT fine-tuning (best-effort)...")
try:
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
except Exception as e:
    print(f"[WARN] GPU clearing (torch) skipped due to: {e}")

# ======================================================================================================================
# 5. DistilBERT Fine-Tuning on CHUNKS

# This section executed the full DistilBERT fine-tuning pipeline. I loaded a base or DAPT-adapted DistilBERT checkpoint,
# tokenized chunks, prepared HuggingFace datasets, defined training arguments, and trained the classifier for 3–5 epochs.
# After training, I evaluated the model, generated metrics, printed a classification report, saved a confusion matrix,
# and wrote both the model and tokenizer to disk for deployment in RegPolicyBot.
# ======================================================================================================================
bert_probs = None                                                                       # Fillup after prediction
if RUN_TRANSFORMER_FINETUNE and HAS_SENTENCE_TRANSFORMERS:
    print("[INFO] Fine-tuning DistilBERT classification model on chunks...")

    dapt_path = os.path.join(MODELS_DIR, "dapt_distilbert")
    base_model_name = dapt_path if os.path.exists(dapt_path) else "distilbert-base-uncased"
    tokenizer_bert = DistilBertTokenizerFast.from_pretrained(base_model_name)
    bert_model = DistilBertForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=num_classes,
    )

    train_texts = df_chunks.loc[train_idx, "text_chunk"].tolist()                       # Building train/test datasets from CHUNK SPLIT
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
        num_train_epochs=5,
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
    metrics_path = os.path.join(RESULTS_DIR, "bert_finetuned_gpu_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(bert_eval, f, indent=4)

    print(f"[INFO] Saved GPU BERT metrics: {metrics_path}")

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

    bert_logits = predictions.predictions                                                 # probabilities for potential ensemble
    bert_probs = softmax(bert_logits, axis=1)
    bert_model.save_pretrained(os.path.join(MODELS_DIR, "bert_finetuned"))                # Save model + tokenizer
    tokenizer_bert.save_pretrained(os.path.join(MODELS_DIR, "bert_tokenizer"))

    print("\n[INFO] Saved fine-tuned DistilBERT model and tokenizer.")
    print("      Model dir:", os.path.join(MODELS_DIR, "bert_finetuned"))
    print("      Tokenizer dir:", os.path.join(MODELS_DIR, "bert_tokenizer"))

else:
    print(
        "[INFO] Skipped DistilBERT fine-tuning "
        "(RUN_TRANSFORMER_FINETUNE=False or sentence-transformers unavailable)."
    )
print("\n[SUMMARY] 4b_modeling_finetune_bert_gpu.py completed.")
# ======================================================================================================================
# SUMMARY OF 4b_modeling_finetune_bert_gpu.py
# ======================================================================================================================
"""
This standalone script successfully completed GPU-accelerated fine-tuning of the DistilBERT classifier used in
RegPolicyBot. It replicated the chunk construction process from 4_modeling.py, produced a clean and deduplicated
chunk dataset, and trained a transformer model capable of significantly outperforming the CPU-trained version.

By isolating the fine-tuning component, this script reduced total runtime, enabled rapid experimentation, and achieved
higher accuracy and F1_macro scores across all categories. The resulting model and tokenizer were saved into the
/models directory, while evaluation artifacts were stored inside /results for downstream reporting and Streamlit
integration.

This GPU-optimized training step strengthened the overall performance of the RegPolicyBot classification pipeline.
"""
# ======================================================================================================================
# REFERENCES
# ======================================================================================================================

# [1] Hugging Face Transformers — Trainer, Tokenizers, Fine-Tuning
#     https://huggingface.co/docs/transformers/main_classes/trainer

# [2] DistilBERT Paper — Distillation of BERT for smaller, faster transformers
#     https://arxiv.org/abs/1910.01108

# [3] Chunking Long Documents for Transformer Classification (ACL 2021)
#     https://aclanthology.org/2021.acl-long.36/

# [4] Scikit-learn Train/Test Split (Stratification Guidelines)
#     https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html

# [5] NLTK Punkt Sentence Tokenizer Documentation
#     https://www.nltk.org/api/nltk.tokenize.html

# [6] SHAP & Confusion Matrix Visualization Standards (Matplotlib)
#     https://matplotlib.org/stable/gallery/images_contours_and_fields/confusion_matrix.html

# [7] Softmax Probability Conversion — SciPy Special Functions
#     https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.softmax.html

# These sources informed the GPU training logic, chunking strategy, transformer architecture,
# evaluation methodology, and best practices for saving and structuring NLP model artifacts.
