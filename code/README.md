## CODE DIRECTORY - RegPolicyBot FINAL PROJECT

#### Project: RegPolicyBot – NLP Final Project
#### Author: Tambudzai Gundani
#### Date: 2025-11-15


This directory houses all executable code for the **RegPolicyBot** project, developed as the final assignment for *DATS 6312 – NLP for Data Science* at The George Washington University.

Each script addresses a specific stage of the project’s NLP pipeline.

| File                                 | Description                                                                                                                                             |
|--------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------|
| **1_setup_folders.py**               | Creates and verifies the core project directories. Establishes a clean structure separating raw data, cleaned data, models, results, and reports.       |
| **2_data_collection.py**             | Connects to the Regulatory Studies Center (RSC) website and retrieves public commentaries, papers, and event transcripts. Saves outputs in `data_raw/`. |
| **3_data_cleaning.py**               | Performs text normalization, removes HTML, punctuation, and stopwords, and applies lemmatization using *spaCy*. Outputs stored in `data_clean/`.        |
| **4_modeling.py**                    | Trains and evaluates multiple NLP models: a TF-IDF + Logistic Regression baseline, an LSTM classifier, and a Transformer + RAG architecture.            |
| **4b_modeling_finetune_bert_gpu.py** | Fine-tunes BERT-based RAG model on RSC corpus                                                                                                           |
| **5_streamlit_app.py**               | Deploys the **RegPolicyBot** prototype using Streamlit, allowing interactive queries and displaying retrieved sources.                                  |
| **6_corpus_app.py**                  | Generates corpus statistics and visualizations                                                                                                          |

### Project Description
RegPolicyBot is an explainable NLP system designed to help users explore research and commentary published by the GW Regulatory Studies Center. It demonstrates multiple natural-language modeling techniques; rule-based, sequential, and transformer based combined with explainability methods such as SHAP and LIME.


