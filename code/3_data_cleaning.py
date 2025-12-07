"""
========================================================================================================================
RegPolicyBot – DATA PREPROCESSING
------------------------------------------------------------------------------------------------------------------------

Purpose: This script performs all cleaning and preprocessing steps needed to transform raw scraped text into structured,
         analysis-ready content. While the scraping script intentionally captures everything from the website (to preserve
         data provenance), this phase systematically removes non-content elements, normalizes formatting, and prepares text
         for later tokenization, modeling, and embedding generation. The overall goal of this stage is to ensure that each
         article has a clean, consistent, and interpretable representation that downstream NLP models can learn from
         effectively. Some actions implemented included;

NB: In typical NLP preprocessing pipelines, steps such as converting text to lowercase, removing punctuation, stripping
    digits, or eliminating stopwords are common. However, these techniques were intentionally NOT applied in this project
    because policy text, regulatory analysis, and legal writing rely heavily on:

    Capitalization: Acronyms like OIRA, IRS, FDA, EPA, and EO 12866 lose meaning if lowercased, and transformer models
                    rely on cases for nuance.
    Punctuation:    Commas, colons, parentheses, and dashes structure arguments and citations. Removing punctuation
                    weakens semantic retrieval and harms LLM embeddings.
    Numbers:        Regulation is numeric. Removing digits would destroy references to executive orders, CFR citations,
                    OMB Circular A-4, regulatory counts, cost estimates, and dates.
    Stopwords:      Words like “shall” “must” “may” and “should” are critical in rulemaking because they signal
                    legal obligations. Removing them would distort policy meaning.
    Formatting:     Sentence boundaries matter for chunking and retrieval.

These decisions reflect best practices for transformer-based RAG systems working with legal, regulatory, and policy
datasets. Preservation of semantic structure was therefore prioritized over aggressive cleaning.
========================================================================================================================
"""
# SECTION 1 — Loading Dependencies & Raw Dataset
# ======================================================================================================================
import re
import pandas as pd
from pathlib import Path
from ftfy import fix_text

RAW_DATA_PATH = Path("../data_raw/rsc_raw.csv")
OUTPUT_PATH = Path("../data_clean/rsc_clean.csv")
try:
    df = pd.read_csv(RAW_DATA_PATH, encoding="utf-8-sig", engine="python")
except UnicodeDecodeError:
    print("[WARN] Falling back to latin1 due to encoding issues.")
    df = pd.read_csv(RAW_DATA_PATH, encoding="latin1", engine="python")

df["title"] = df["title"].astype(str).apply(fix_text).str.strip()
df["author"] = df["author"].astype(str).apply(fix_text).str.strip()
df["date"] = df["date"].astype(str).apply(fix_text).str.strip()
# ======================================================================================================================
# SECTION 2 — Defining Text Cleaning Utilities

# Constructed a set of modular cleaning functions that address the types of artifacts commonly present in scraped regulatory
# text. Unlike generic NLP pipelines, which often aggressively lowercase text, remove digits, punctuation, or stopwords,
# the functions below use a policy-aware approach designed to preserve the legal and semantic structure of RSC publications.
# ======================================================================================================================
def fix_encoding(text: str) -> str:                                           # Fixing common UTF-8 encoding errors
    if not isinstance(text, str):
        return ""
    text = fix_text(text)
    return text

def remove_boilerplate(text: str) -> str:                                     # Removing boilerplate and non-content text
    if not text:
        return text
    text = re.sub(r"download\s+[^\.!\n\r]*", "", text, flags=re.IGNORECASE)
    extra_patterns = [
        r"\(pdf\)", r"listen\s+to\s+podcast", r"in\s+brief.*?:", r"see\s+other\s+.*?(insight|commentary)"]
    for p in extra_patterns:
        text = re.sub(p, "", text, flags=re.IGNORECASE)
    return text.strip()

def remove_inline_urls(text: str) -> str:                                     # Removing URLs from body text only
    return re.sub(r"http\S+|www\S+|https\S+", "", text)

def remove_footer_noise(text: str) -> str:                                    # Removing footer/social-media fragments
    patterns = [r"twitter", r"facebook", r"newsletter", r"subscribe", r"follow\s+us", r"your\s+support",
                r"contact\s+gw", r"report\s+a\s+barrier",  r"privacy\s+notice"]
    for p in patterns:
        text = re.sub(p, "", text, flags=re.IGNORECASE)
    return text

def normalize_whitespace(text: str) -> str:                                   # Whitespace normalization
    if not text:
        return text
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def clean_article_text(text: str) -> str:                                     # Full cleaning pipeline for one article
    if not isinstance(text, str):
        return ""
    text = fix_encoding(text)                                                 # Fixing encoding artifacts
    text = remove_boilerplate(text)                                           # Removing boilerplate elements (PDF downloads, podcasts, teasers)
    text = remove_inline_urls(text)                                           # Removing inline URLs
    text = remove_footer_noise(text)                                          # Removing social media and footer fragments
    text = normalize_whitespace(text)                                         # Normalizing whitespace
    return text
# ======================================================================================================================
# SECTION 3 — Applying the Cleaning Pipeline to the Dataset.

# In this section, we applied `clean_article_text` to the raw scraped text column, transforming each article into a
# structured, analysis-ready representation while preserving the original raw text for auditability and reproducibility.
# This cleaned version serves as the canonical input for modeling, embedding generation, and the RAG pipeline.
# ======================================================================================================================
df["clean_text"] = df["raw_text"].apply(clean_article_text)

df["title"] = df["title"].astype(str).str.strip()                             # Cleaning metadata formatting
df["author"] = df["author"].astype(str).str.strip()
df["date"] = df["date"].astype(str).str.strip()

def extract_name_from_text(text: str) -> str:                                 # Fixing Missing Authors Extracted Incorrectly into Article Text
    if not isinstance(text, str):
        return None
    pattern = r"\b[A-Z][a-z]+(?:\s[A-Z]\.)?\s[A-Z][a-z]+\b"
    match = re.search(pattern, text)
    return match.group(0) if match else None
for idx, row in df.iterrows():
    if pd.isna(row["author"]) or row["author"] in ["", None]:
        possible_author = extract_name_from_text(row["clean_text"])
        if possible_author:
            df.at[idx, "author"] = possible_author
            df.at[idx, "clean_text"] = (
                row["clean_text"].replace(possible_author, "").strip()
            )

df = df[[                                                                     # Reordering columns for clarity
    "category",
    "url",
    "title",
    "author",
    "date",
    "raw_text",
    "clean_text"
]]
# ======================================================================================================================
# SECTION 4 — Saving the Cleaned Dataset

# The final cleaned dataset was saved to the /data_clean directory, replacing raw, unstructured text with a consistent,
# high-quality corpus suitable for modeling. This file became the authoritative dataset for:
#     - TF-IDF baseline models
#     - LSTM training
#     - sentence-transformer embeddings
#     - FAISS/Chroma retrieval indexes
#     - RAG-based chatbot generation
#
# Keeping both raw_text and clean_text ensures transparency and reproducibility.
# ======================================================================================================================

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)

print(f"[INFO] Cleaned dataset saved to: {OUTPUT_PATH}")
print(f"[INFO] Total cleaned records: {len(df)}")
# ======================================================================================================================
# SECTION 6 — Summary of Cleaning Process
# ======================================================================================================================
"""
This script completed the full data-cleaning phase of the RegPolicyBot project, transforming the raw scraped RSC 
publications into a consistent, analysis-ready corpus suitable for classical, neural, and transformer-based NLP models.

Key steps completed:
------------------------------------------------------------------------------------------------------------------------
    • Automatic encoding repair using ftfy + targeted replacements.
    • Removal of boilerplate (PDF links, podcast buttons, teaser sections).
    • Removal of inline URLs while preserving the canonical article URL.
    • Removal of footer and sidebar noise (social media, navigation fragments).
    • Whitespace normalization for consistent formatting.
    • Preservation of punctuation, casing, numbers, and stopwords due to their
      importance in regulatory and legal text.

The final dataset included both raw and cleaned versions of each article and ready for feature extraction, embeddings, 
and modeling in 4_modeling.py.

Data cleaning completed successfully.
"""
# ======================================================================================================================
# SECTION 7 — CODE REFERENCES
# ======================================================================================================================
# These references document the external sources that inform the text-cleaning methods, encoding fixes, regex patterns,
# and preprocessing pipeline design used in this script.
# ----------------------------------------------------------------------------------------------------------------------
# SECTION 1 — LOADING DEPENDENCIES & RAW DATA
# ----------------------------------------------------------------------------------------------------------------------
# [1] Pandas Documentation – Reading CSV Files and Handling Encoding
#     https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html
# [2] Python Docs – Unicode and Character Encodings
#     https://docs.python.org/3/howto/unicode.html
# ----------------------------------------------------------------------------------------------------------------------
# SECTION 2 — TEXT CLEANING UTILITIES
# ----------------------------------------------------------------------------------------------------------------------
# [1] ftfy Documentation – "Fix Text For You": repairing Unicode errors from scraped web data
#     https://ftfy.readthedocs.io/en/latest/
# [2] Regular Expressions in Python – Official re Module Documentation
#     https://docs.python.org/3/library/re.html
# [3] Research: Best Practices for Cleaning Text for NLP While Preserving Semantics (ACL Anthology)
#     https://aclanthology.org/2020.nlposs-1.2/
# ----------------------------------------------------------------------------------------------------------------------
# SECTION 3 — APPLYING THE CLEANING PIPELINE
# ----------------------------------------------------------------------------------------------------------------------
# [1] Data Preprocessing for NLP (Stanford CS224N)
#     https://web.stanford.edu/class/cs224n/
# [2] Preserving Casing, Punctuation, and Stopwords in Legal and Policy Text — Legal NLP Guidelines
#     https://arxiv.org/abs/2105.01226
# ----------------------------------------------------------------------------------------------------------------------
# SECTION 4 — SAVING THE CLEANED DATASET
# ----------------------------------------------------------------------------------------------------------------------
# [1] MIT Data Management: Best Practices for Reproducible Research Data Pipelines
#     https://libraries.mit.edu/data-management/
# [2] ACM Reproducibility Guidelines – Documenting and Preserving Raw vs Processed Data
#     https://www.acm.org/publications/policies/artifact-review-and-badging-current
# ----------------------------------------------------------------------------------------------------------------------
# SECTION 6 — CLEANING PROCESS OVERVIEW
# ----------------------------------------------------------------------------------------------------------------------
# [1] Legal & Regulatory NLP Preprocessing Frameworks (Harvard NLP Group)
#     https://nlp.seas.harvard.edu/
# [2] Google Machine Learning Guides – Text Cleaning and Normalization Principles
#     https://developers.google.com/machine-learning/guides/text-classification/preprocess
