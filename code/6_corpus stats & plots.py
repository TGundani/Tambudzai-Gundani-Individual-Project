# ==========================================================
# 6_visualizations.py
# Unified visualization pipeline for RegPolicyBot
# ==========================================================
# Outputs:
#   results/rsc_verified_corpus_statistics.csv
#   results/distilbert_loss_curve.png
#   results/lime_example_real.html
#   results/shap_summary_logreg.png
# ==========================================================

import os
import pandas as pd
import matplotlib.pyplot as plt
import json
from joblib import load

# ----------------------------------------------------------
# 0. SETUP — dynamic paths (no hardcoding)
# ----------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data_clean")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ----------------------------------------------------------
# 1. CORPUS STATISTICS
# ----------------------------------------------------------
print("\n[INFO] Computing corpus statistics...")

clean_path = os.path.join(DATA_DIR, "rsc_clean.csv")
chunks_path = os.path.join(DATA_DIR, "df_chunks.csv")

clean = pd.read_csv(clean_path)
chunks = pd.read_csv(chunks_path)

# Word counts
clean["word_count"] = clean["clean_text"].astype(str).apply(lambda x: len(x.split()))
doc_stats = clean.groupby("category")["word_count"].agg(
    Documents="count", Total_Words="sum", Avg_Words_per_Doc="mean"
).reset_index()

# Chunk counts
chunks["n_tokens"] = chunks["text_chunk"].astype(str).apply(lambda x: len(x.split()))
chunk_stats = chunks.groupby("orig_category")["n_tokens"].agg(
    Chunks="count", Avg_Tokens_per_Chunk="mean"
).reset_index().rename(columns={"orig_category": "category"})

# Merge
corpus_stats = pd.merge(doc_stats, chunk_stats, on="category", how="outer")
total_row = pd.DataFrame({
    "category": ["Total"],
    "Documents": [corpus_stats["Documents"].sum()],
    "Total_Words": [corpus_stats["Total_Words"].sum()],
    "Avg_Words_per_Doc": [corpus_stats["Avg_Words_per_Doc"].mean()],
    "Chunks": [corpus_stats["Chunks"].sum()],
    "Avg_Tokens_per_Chunk": [corpus_stats["Avg_Tokens_per_Chunk"].mean()]
})
corpus_stats = pd.concat([corpus_stats, total_row], ignore_index=True)

print("\n=== VERIFIED RSC CORPUS STATISTICS ===")
print(corpus_stats.to_string(index=False))
corpus_stats.to_csv(os.path.join(RESULTS_DIR, "rsc_verified_corpus_statistics.csv"), index=False)

# ----------------------------------------------------------
# 2. DISTILBERT TRAINING LOSS CURVE
# ----------------------------------------------------------
print("\n[INFO] Plotting DistilBERT loss curve...")

trainer_state_path = os.path.join(MODELS_DIR, "distilbert_classifier", "checkpoint-2295", "trainer_state.json")

if os.path.exists(trainer_state_path):
    with open(trainer_state_path) as f:
        state = json.load(f)

    losses = [l["loss"] for l in state["log_history"] if "loss" in l]
    evals = [l["eval_loss"] for l in state["log_history"] if "eval_loss" in l]
    steps = list(range(1, len(losses) + 1))

    plt.figure(figsize=(6, 4))
    plt.plot(steps, losses, label="Training Loss")
    plt.plot(steps[:len(evals)], evals, label="Validation Loss")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("DistilBERT Fine-Tuning Loss Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "distilbert_loss_curve.png"), dpi=300)
    plt.close()
    print("✓ Saved: results/distilbert_loss_curve.png")
else:
    print("✗ trainer_state.json not found — skipping loss curve.")

# ----------------------------------------------------------
# 3. LIME EXPLANATION (REAL SAMPLE)
# ----------------------------------------------------------
print("\n[INFO] Generating LIME explanation from real RSC sample...")

try:
    from lime.lime_text import LimeTextExplainer
    logreg_path = os.path.join(MODELS_DIR, "logreg_tfidf_model.joblib")
    vectorizer_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib")

    logreg = load(logreg_path)
    vectorizer = load(vectorizer_path)

    # Pick a real sample from RSC dataset
    sample = clean.sample(1, random_state=42)
    sample_text = sample["clean_text"].values[0][:2000]  # limit for readability
    sample_category = sample["category"].values[0]

    explainer = LimeTextExplainer(class_names=logreg.classes_)
    exp = explainer.explain_instance(
        sample_text,
        lambda x: logreg.predict_proba(vectorizer.transform(x)),
        num_features=10
    )
    from IPython.display import display

    fig = exp.as_pyplot_figure()
    fig.savefig(os.path.join(RESULTS_DIR, "lime_example_real.png"), dpi=300, bbox_inches="tight")
    lime_output = os.path.join(RESULTS_DIR, "lime_example_real.html")
    exp.save_to_file(lime_output)
    print(f"✓ Saved: {lime_output}")
    print(f"  Example category: {sample_category}")
except Exception as e:
    print(f"✗ LIME generation failed: {e}")

# ----------------------------------------------------------
# 4. SHAP FEATURE IMPORTANCE (REAL SAMPLE)
# ----------------------------------------------------------
import shap
import numpy as np
import matplotlib.pyplot as plt
from joblib import load

logreg = load(os.path.join(MODELS_DIR, "logreg_tfidf_model.joblib"))
vectorizer = load(os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib"))

texts = clean["clean_text"].sample(100, random_state=123).tolist()
X = vectorizer.transform(texts)

explainer = shap.LinearExplainer(logreg, X, feature_perturbation="interventional")
shap_values = explainer.shap_values(X)

# Explicit figure handle ensures save works in all environments
plt.figure()
shap.summary_plot(
    shap_values,
    feature_names=vectorizer.get_feature_names_out(),
    show=False,
    plot_type="bar"
)
plt.title("Top Word Features by SHAP Importance (Logistic Regression)")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "shap_summary_logreg.png"), dpi=300)
plt.close()
print("✓ Saved: results/shap_summary_logreg.png")

# ----------------------------------------------------------
# DONE
# ----------------------------------------------------------
print("\n[INFO] All visual outputs saved in:", RESULTS_DIR)
print("Files generated:\n",
      "- rsc_verified_corpus_statistics.csv\n",
      "- distilbert_loss_curve.png\n",
      "- lime_example_real.html\n",
      "- shap_summary_logreg.png")
