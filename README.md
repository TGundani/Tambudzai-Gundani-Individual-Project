# RSC RegPolicyBot – NLP Final Project

**Author:** Tambudzai Gundani  
**Course:** DATS 6312 – NLP for Data Science  
**Institution:** The George Washington University  
**Date:** Fall 2025  

---

## 1. Project Overview
**RegPolicyBot** is an explainable AI chatbot that retrieves and summarizes policy insights from the **GWU Regulatory Studies Center (RSC)**.  
It combines classical, recurrent, and transformer-based NLP models within a **Retrieval-Augmented Generation (RAG)** architecture.  
The system demonstrates how transparent NLP methods can make complex regulatory research accessible to the public and policymakers.

---
### The system combines:

- **Classical NLP** – TF-IDF + Logistic Regression for topic classification  
- **Recurrent networks** – LSTM for capturing sequential context  
- **Transformer-based retrieval and generation** – sentence embeddings + small open-weight LLM in a  
  Retrieval-Augmented Generation (**RAG**) pipeline  
- **Explainability tools** – SHAP / LIME + simple autoencoder for topic clusters  

### Goal: 
Make complex regulatory research more **searchable, interpretable, and interactive** for policymakers,  
students, journalists, and the public by answering natural-language questions with **concise, cited,  
evidence-linked responses** from RSC publications.
---
## 2. Repository Structure

The main project directory is organized to mirror the full NLP workflow  
**(data acquisition → preprocessing → modeling → evaluation → deployment).**

---
## Submitted For Grading in Github 
- **Individual-Final-Project-Report/**
- **Individual-Final-Project-Presentation/**
- **Code/**
- **README.md**

 **NB: All Other Folders Due to Size have been Uploaded to Goodgle Drive:**
https://drive.google.com/drive/folders/1GzEgADFgwf86zxIwcN_nOgVBiWMnipQI?usp=sharing

```text

FinalProject_RegPolicyBot/
├── README.md                     # Overall project documentation (this file)
│
├── code/                         # Source code (run in numeric order)
│   ├── 1_setup_folders.py        # Creates project directories and base config
│   ├── 2_data_collection.py      # Scrapes / loads RSC publications and metadata
│   ├── 3_data_cleaning.py        # Cleans, tokenizes, lemmatizes, and chunks text
│   ├── 4_modeling.py             # Builds & evaluates baseline, LSTM, and RAG models
│   ├── 4b_modeling_finetune_bert # Fine-tunes BERT-based RAG model on RSC corpus          
│   ├── 5_streamlit_app.py        # Runs the RegPolicyBot Streamlit prototype
│   ├── 6_corpus stats & plots.py # Generates corpus statistics and visualizations           
│   └── README.md                 # Script-level usage notes (arguments, examples)
│
├── data_raw/                     # Unprocessed scraped / downloaded data
│   └── rsc_raw.csv               # Raw corpus (one row per document or chunk)
│
├── data_clean/                   # Cleaned and lemmatized data
│   └── rsc_clean.csv             # Clean text + chunk labels / topics
│
├── models/                       # Saved models and embedding indexes
│   ├── baseline_logreg.pkl       # TF-IDF + Logistic Regression classifier
│   ├── lstm_model.h5             # Trained LSTM model weights
│   └── vector_index/             # FAISS or Chroma index for retrieval
│
├── results/                      # Evaluation metrics and visualizations
│   ├── performance_metrics.csv   # Accuracy, F1, BLEU, ROUGE, Recall@k, etc.
│   ├── shap_visualizations/      # SHAP / LIME plots for model explainability
│   └── topic_clusters.png        # 2D projection of autoencoder topic clusters
│
├── reports/                      # Final report & presentation
    ├── RegPolicyBot_Final_Report.pdf
    └── RegPolicyBot_Presentation_Slides.pptx
```