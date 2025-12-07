"""
========================================================================================================================
RegPolicyBot – STREAMLIT INTERFACE (HYBRID RAG + CLASSIFIER)
------------------------------------------------------------------------------------------------------------------------
Purpose:
    This Streamlit application served as the final user-facing interface for the RegPolicyBot system. It combined
    multiple NLP components developed earlier in the pipeline — a fine-tuned DistilBERT classifier, a MiniLM-based
    semantic retrieval backend, and an optional OpenAI LLM — into a unified regulatory intelligence assistant.

What the application accomplished:
    • Loaded structured regulatory metadata used for analytics-style queries.
    • Loaded all model artifacts created in 4_modeling.py and 4b_modeling_finetune_bert_gpu.py.
    • Classified user questions by RSC category using a fine-tuned DistilBERT model.
    • Retrieved the top-k semantically similar RSC documents using MiniLM embeddings + NearestNeighbors.
    • Generated responses either:
         – using a heuristic offline summarizer, or
         – using an OpenAI LLM prompted with retrieved context.
    • Detected metadata-related queries and returned tables, summaries, and charts.
    • Provided an interactive interface for regulatory research, demonstrating a full hybrid RAG workflow.

Why this interface was needed:
    • It represented the deployment layer of the project.
    • It integrated classification, retrieval, metadata analytics, visualization, and LLM-based reasoning.
    • It demonstrated RAG concepts in a live, reproducible system suitable for policy analysis.

Outputs:
    • Real-time category prediction
    • Top-k retrieved RSC documents
    • Metadata tables and summaries
    • Auto-generated charts
    • LLM-based or heuristic final answers

This file completed the end-to-end RegPolicyBot pipeline by making the system interactive.
"""
# ======================================================================================================================
# Importing Libraries
# ======================================================================================================================
import os
import re
import json
import torch
import base64
import joblib
import textwrap
import numpy as np
import pandas as pd
import streamlit as st
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import matplotlib.pyplot as plt
from scipy.special import softmax
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors

try:
    from openai import OpenAI
    HAS_OPENAI = True
except Exception:
    HAS_OPENAI = False
# ======================================================================================================================
#  SECTION 1 — Loading Metadata CSVs

# This section loaded structured regulatory metadata files containing Federal Register statistics, major rule counts, and
# rulemaking activity by the presidential administration. These datasets were used to answer analytics-style queries whenever
# a user asked about regulatory trends instead of document-based RAG queries.
# ======================================================================================================================
@st.cache_data
def load_metadata():
    fr_tracking = pd.read_csv("../backend_metadata/fr_tracking.csv",encoding="latin1", engine="python")
    major_rules = pd.read_csv("../backend_metadata/major_rules_by_presidential_year.csv",encoding="latin1", engine="python")
    fr_by_pres = pd.read_csv("../backend_metadata/federal_register_rules_by_presidential_year.csv",encoding="latin1", engine="python")
    return fr_tracking, major_rules, fr_by_pres
fr_tracking, major_rules, fr_by_pres = load_metadata()

def get_base64_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()
banner_img = get_base64_image("../assets/RSC.png")

st.markdown(f"""
<style>

/* --- Top Banner Image (No Fade) --- */
.top-banner {{
    width: 100%;
    height: 220px;
    background-image: url("data:image/png;base64,{banner_img}");
    background-size: cover;
    background-position: center top;
    background-repeat: no-repeat;
    border-bottom: 3px solid #004E8C;
}}

.banner-spacing {{
    height: 25px;
}}

</style>
""", unsafe_allow_html=True)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_CLEAN_DIR = os.path.join(BASE_DIR, "data_clean")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

BERT_MODEL_DIR = os.path.join(MODELS_DIR, "bert_finetuned")
BERT_TOKENIZER_DIR = os.path.join(MODELS_DIR, "bert_tokenizer")
LABEL_ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder.joblib")

MINILM_EMB_PATH = os.path.join(MODELS_DIR, "rsc_miniLM_embeddings.npy")
NN_INDEX_PATH = os.path.join(MODELS_DIR, "rsc_embeddings_nn_index.joblib")
RETRIEVAL_MANIFEST_PATH = os.path.join(RESULTS_DIR, "rsc_retrieval_manifest.csv")

# ----------------------------------------------------------------------------------------------------------------------
# Streamlit page config
# ----------------------------------------------------------------------------------------------------------------------
st.markdown('<div class="top-banner"></div>', unsafe_allow_html=True)
st.markdown('<div class="banner-spacing"></div>', unsafe_allow_html=True)

st.set_page_config(
    page_title="RegPolicyBot – Regulatory Studies Center",
    page_icon="⚖️",
    layout="wide",
)
# ----------------------------------------------------------------------------------------------------------------------
# Cached loaders
# ----------------------------------------------------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_label_encoder():
    return joblib.load(LABEL_ENCODER_PATH)

@st.cache_resource(show_spinner=True)
def load_bert_classifier():
    """
    Load fine-tuned DistilBERT classifier + tokenizer + label encoder.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = DistilBertTokenizerFast.from_pretrained(BERT_TOKENIZER_DIR)
    model = DistilBertForSequenceClassification.from_pretrained(BERT_MODEL_DIR)
    model.to(device)
    model.eval()
    label_encoder = load_label_encoder()
    return tokenizer, model, label_encoder, device

@st.cache_resource(show_spinner=True)
def load_retrieval_backend():
    """
    Load MiniLM sentence-transformer, embeddings matrix, NN index, and manifest.
    """
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    sent_model = SentenceTransformer(model_name)
    embeddings = np.load(MINILM_EMB_PATH)
    nn_index: NearestNeighbors = joblib.load(NN_INDEX_PATH)
    manifest = pd.read_csv(RETRIEVAL_MANIFEST_PATH)
    return sent_model, embeddings, nn_index, manifest

@st.cache_resource(show_spinner=False)
def get_openai_client(api_key: str | None):
    """
    Create OpenAI client if library and API key are available.
    """
    if not HAS_OPENAI or not api_key:
        return None
    client = OpenAI(api_key=api_key)
    return client
# ======================================================================================================================
# Helper functions

# This group of functions handled core RAG behaviors. The classifier predicted the RSC category of the user question, the
# retrieval function used MiniLM embeddings + NearestNeighbors to find top-k relevant RSC documents, and the context
# builder assembled retrieved snippets into a structured block for LLM prompting.
# ======================================================================================================================

def classify_question(text: str):
    """
    Use fine-tuned DistilBERT to classify the user query into an RSC category.
    Returns (pred_label_str, pred_prob, probs_dict).
    """
    tokenizer, model, label_encoder, device = load_bert_classifier()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256,
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits.cpu().numpy()

    probs = softmax(logits, axis=1)[0]
    pred_idx = int(np.argmax(probs))
    pred_label = label_encoder.inverse_transform([pred_idx])[0]
    pred_prob = float(probs[pred_idx])

    probs_dict = {
        label_encoder.inverse_transform([i])[0]: float(p)
        for i, p in enumerate(probs)
    }

    return pred_label, pred_prob, probs_dict


def retrieve_documents(query: str, top_k: int = 5):
    """
    Use MiniLM embeddings + NearestNeighbors index to retrieve top-k docs.
    Returns a list of dict rows with manifest metadata + similarity score.
    """
    sent_model, embeddings, nn_index, manifest = load_retrieval_backend()

    query_emb = sent_model.encode([query], normalize_embeddings=True)
    distances, indices = nn_index.kneighbors(query_emb, n_neighbors=top_k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        row = manifest.iloc[idx].to_dict()
        row["doc_index"] = int(idx)
        row["cosine_similarity"] = float(1 - dist)
        results.append(row)

    return results


def build_context_block(retrieved_docs, max_chars: int = 4000) -> str:
    """
    Build a textual context block from retrieved documents for the LLM.
    """
    pieces = []
    for i, doc in enumerate(retrieved_docs, start=1):
        title = doc.get("title", f"Doc {i}")
        category = doc.get("category", "")
        url = doc.get("url", "")
        year = doc.get("document_year", "")
        snippet = doc.get("text_preview", "") if "text_preview" in doc else ""

        # If manifest did not preserve text_preview, at least show title + url
        chunk = f"[{i}] Title: {title}\nCategory: {category} | Year: {year}\nURL: {url}\nSnippet: {snippet}\n"
        pieces.append(chunk)

    context = "\n\n".join(pieces)
    if len(context) > max_chars:
        context = context[:max_chars] + "\n\n[Context truncated for length...]"
    return context

# ======================================================================================================================
#  SECTION 2 — Detecting Metadata Questions

# In this section, I implemented a keyword-driven detector that identified whether a user question referred to structured
# regulatory metadata rather than unstructured RSC text. If matched, the system routed the query to metadata tables,
# bypassing classification and retrieval entirely.
# ======================================================================================================================
def detect_metadata_question(user_q):
    q = user_q.lower()
    keywords = [
        "major rule", "major rules",
        "federal register", "fr tracking",
        "significant rule", "econ significant",
        "economically significant", "significant rules",
        "regulatory pages", "pages",
        "administration", "president",
        "rulemaking output", "rules issued"
    ]
    return any(kw in q for kw in keywords)
# ======================================================================================================================
#  SECTION 3 — Metadata Answer Generator

#  This section generated structured-data responses for metadata queries. It formatted tables using Markdown, computed
#  summaries, and provided contextual explanations. These outputs allowed the system to behave like a regulatory analytics
#  assistant when no text-based retrieval was needed.
# ======================================================================================================================
def answer_with_metadata(user_q):
    q = user_q.lower()
    # ------------------------- 1: Major Rules Data ---------------------
    if "major rule" in q:
        df = major_rules.copy()
        table = df.to_markdown(index=False)
        count_col = "major_rules_published" if "major_rules_published" in df.columns else "major_rules_count"
        summary = (
            "The dataset shows major rules issued by each administration over time. "
            f"For example, the highest year appears to be {df[count_col].max()} major rules, "
            f"and the lowest is {df[count_col].min()}."
        )
        return f"### Major Rules by Presidential Year\n\n{table}\n\n**Summary:** {summary}"

    # ----------------------- 2: Federal Register Summary by Year --------------
    if "federal register" in q or "fr tracking" in q:
        df = fr_tracking.copy()
        table = df.head(50).to_markdown(index=False)
        summary = (
            f"This Federal Register tracking dataset includes {len(df)} entries. "
            f"The top issuing agency appears to be **{df['agency'].mode()[0]}**. "
            f"There are {df['significant'].sum()} significant rules and "
            f"{df['econ_significant'].sum()} economically significant rules recorded."
        )
        return f"### Federal Register Tracking\n\n{table}\n\n**Summary:** {summary}"

    # -----------------------3: Rules by President ------------------------------
    if "president" in q and "rules" in q:
        df = fr_by_pres.copy()
        table = df.to_markdown(index=False)
        summary = (
            f"The dataset shows rulemaking output by each administration. "
            f"The president with the highest number of total rules appears to be "
            f"{df.loc[df['total_rules'].idxmax(), 'president']}."
        )
        return f"### Rules by President\n\n{table}\n\n**Summary:** {summary}"

    return None
# ======================================================================================================================
#  SECTION 4 — Metadata Filtering Engine

#  This section extracted filters (agency, year, significance) from natural-language questions and applied them to the
#  Federal Register tracking dataset. It returned filtered tables and summaries to support more detailed data exploration
#  based on user intent.
# ======================================================================================================================
def filter_metadata(user_q):
    q = user_q.lower()
    df = fr_tracking.copy()

    # --------------------------- 1. YEAR DETECTION -----------------------------
    year_match = re.search(r"(19|20)\d{2}", q)
    year = None
    if year_match:
        year = year_match.group(0)
        df = df[df["publication_date"].str.contains(year, na=False)]

    # -------------------------- 2. AGENCY DETECTION ----------------------------
    agencies = df["agency"].dropna().unique()
    agency_match = None
    for ag in agencies:
        if ag.lower() in q:
            agency_match = ag
            df = df[df["agency"] == ag]
            break

    # ------------------------- 3. SIGNIFICANCE FILTER --------------------------
    if "significant" in q:
        df = df[df["significant"] == 1]
    if "economically significant" in q or "econ significant" in q:
        df = df[df["econ_significant"] == 1]

    # -------------------------- 4. RETURN RESULT -------------------------------
    if df.empty:
        return None, None, "I couldn't find matching entries."
    table = df.head(50).to_markdown(index=False)
    summary = f"Found **{len(df)}** matching entries."
    if year:
        summary += f" Filtered by year **{year}**."
    if agency_match:
        summary += f" Filtered by agency **{agency_match}**."
    if "significant" in q:
        summary += " Showing only significant rules."
    if "economically significant" in q:
        summary += " Showing only economically significant rules."
    return df, table, summary

# ======================================================================================================================
#  SECTION 5 — Chart Generator

# This section created interactive charts in response to metadata queries requesting visualizations. It generated bar and
# line plots for major rules and significant rules over time, enabling quick graphical interpretation of regulatory trends.
# ======================================================================================================================
def try_chart(user_q):
    q = user_q.lower()

    # ------------------------------ 1. Major Rules by Year Chart ------------------
    if "major" in q and "chart" in q:
        df = major_rules.copy()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(df["year"], df["major_rules_count"])
        ax.set_title("Major Rules by Year")
        ax.set_xlabel("Year")
        ax.set_ylabel("Major Rules Issued")
        return fig, "Chart: Major rules issued by year."

    # ---- -------------------------2. Significant Rules Chart -----------------------
    if "significant" in q and "chart" in q:
        df = fr_tracking.copy()
        df["year"] = df["publication_date"].str[-4:]
        agg = df.groupby("year")["significant"].sum().reset_index()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(agg["year"], agg["significant"])
        ax.set_title("Significant Rules Over Time")
        ax.set_xlabel("Year")
        ax.set_ylabel("Count of Significant Rules")
        return fig, "Chart: Significant rules over time."
    return None, None

def generate_heuristic_answer(question: str, category: str, retrieved_docs):
    """
    Offline, a non-LLM answer generator that summarizes retrieved docs.
    """
    if not retrieved_docs:
        return (
            "I could not retrieve any relevant RSC documents for this question. "
            "Please try rephrasing or narrowing your query."
        )

    top_titles = [d.get("title", f"Document {i+1}") for i, d in enumerate(retrieved_docs)]
    bullets = "\n".join(f"- {t}" for t in top_titles)

    text = f"""
Based on your question, this looks most related to RSC materials in the
**{category}** category.
Here are some key RSC documents that appear most relevant:
{bullets}
I recommend exploring these documents for detailed discussion. The chatbot retrieval
backend finds them by semantic similarity using MiniLM sentence embeddings and
the RSC retrieval index.
"""
    return textwrap.dedent(text).strip()
# ======================================================================================================================
#  SECTION 6 — LLM Summary for Metadata

#  This section generated policy-relevant summaries using an OpenAI LLM for metadata-filtered queries.When a user provided
#  an API key, the application produced higher-level insights instead of simple table descriptions.
# ======================================================================================================================
def llm_summarize_metadata(df, user_q):
    import openai
    openai.api_key = st.session_state.get("openai_api_key", None)
    if not openai.api_key:
        return None

    prompt = f"""
You are a policy analyst at the GW Regulatory Studies Center.
The user asked: "{user_q}"
Based on this metadata table of regulatory actions:
{df.head(20).to_markdown()}
Write a concise, insightful, policy-relevant summary of what the data shows.
"""
    try:
        resp = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=250
        )
        return resp.choices[0].message["content"]
    except:
        return None
def generate_llm_answer(
    client,
    question: str,
    category: str,
    retrieved_docs,
    model_name: str = "gpt-4o-mini",
):
    """
    Use OpenAI client to synthesize an answer from retrieved context.
    Falls back to heuristic summary if something goes wrong.
    """
    if client is None or not retrieved_docs:
        return generate_heuristic_answer(question, category, retrieved_docs)
    context_block = build_context_block(retrieved_docs)
    system_prompt = (
        "You are RegPolicyBot, a helpful assistant for the GW Regulatory Studies Center. "
        "You answer questions using ONLY the provided RSC context. "
        "Cite documents in a natural way like [Doc 1], [Doc 2]. "
        "If the context is not sufficient, say so explicitly."
    )
    user_prompt = f"""
User question:
{question}
Predicted RSC category: {category}
Relevant RSC documents:
{context_block}
Using ONLY the information above, write a concise, well-structured answer (2–5 paragraphs).
"""
    try:
        resp = client.responses.create(
            model=model_name,
            temperature=0.3,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        answer = resp.output[0].content[0].text
        return answer.strip()
    except Exception as e:
        return (
            "I encountered an issue calling the LLM backend:\n"
            f"`{e}`\n\n"
            "Here is an offline summary based on retrieved documents instead:\n\n"
            + generate_heuristic_answer(question, category, retrieved_docs)
        )
# ======================================================================================================================
# Section 7: UI Layout + Main Interaction

# This final section defined the full Streamlit interface: layout, sidebar configuration, question  submission workflow,
# metadata branching logic, classification display, document retrieval display, and answer generation. This section
# connected all backend components to the interactive frontend used by end-users.
# ======================================================================================================================
st.title("⚖️ RegPolicyBot – GW Regulatory Studies Center")
st.markdown(
    """
This prototype combines:

- A **fine-tuned DistilBERT classifier** that predicts which RSC category your
  question belongs to (Commentaries & Insights, Events, Journal Articles & Working Papers).
- A **MiniLM-based retrieval backend** that finds the most relevant RSC documents.
- An optional **OpenAI LLM** that synthesizes an answer using those retrieved documents.

Use this tool to explore how RAG and classification can support regulatory research and education.
"""
)

with st.sidebar:
    st.header("Model & Retrieval Settings")
    answer_mode = st.radio(
        "Answer mode",
        options=["Heuristic (offline)", "OpenAI LLM (hybrid RAG)"],
        index=1,
        help="Heuristic mode uses only local logic; OpenAI mode calls an LLM using retrieved RSC context.",
    )

    openai_key = None
    llm_client = None
    if answer_mode == "OpenAI LLM (hybrid RAG)":
        st.markdown("#### OpenAI configuration")
        st.caption(
            "You can either set the `OPENAI_API_KEY` environment variable or paste a key below."
        )
        env_key = os.environ.get("OPENAI_API_KEY")
        user_key = st.text_input(
            "OpenAI API key",
            type="password",
            value=env_key or "",
            help="Key is used only in this session to call the OpenAI API.",
        )
        openai_key = user_key.strip() or env_key
        if not openai_key:
            st.warning("No OpenAI API key provided; the app will fall back to heuristic answers.")
        llm_client = get_openai_client(openai_key)
    st.session_state["openai_api_key"] = openai_key
    top_k = st.slider("Number of documents to retrieve", min_value=3, max_value=10, value=5, step=1)
    st.markdown("---")
    st.markdown(
        """
**About this app**

This interface is part of the *RegPolicyBot* final project for the NLP course.
It demonstrates classification, retrieval, and explainable RAG over the
Regulatory Studies Center corpus.
"""
    )

st.subheader("🔍 Ask a question about regulation or RSC work")
default_q = "How has the Regulatory Studies Center contributed to discussions on benefit-cost analysis?"
user_question = st.text_area(
    "Enter your question:",
    value=default_q,
    height=120,
)
col_run, col_clear = st.columns([1, 1])
with col_run:
    run_btn = st.button("Submit")
with col_clear:
    clear_btn = st.button("Clear")
# ======================================================================================================================
# Section 7: LLM Answer Generator (RAG)

# This component synthesized final answers using retrieved RSC text. It constructed a prompt that included the predicted
# category and retrieved context, and invoked an OpenAI LLM to generate a grounded response. If no LLM was available,
# the system fell back to an offline heuristic summarizer.
# ======================================================================================================================
if detect_metadata_question(user_question):
    filtered_df, table, summary = filter_metadata(user_question)
    if table:
        st.markdown("### Filtered Federal Register Results")
        st.markdown(table)

        if st.session_state.get("openai_api_key"):                                  # Adding OpenAI summary if available
            llm_summary = llm_summarize_metadata(filtered_df, user_question)
            if llm_summary:
                st.markdown("### AI Summary")
                st.write(llm_summary)
        else:
            st.markdown(f"**Summary:** {summary}")

        fig, chart_summary = try_chart(user_question)                               # Checking if charts requested
        if fig:
            st.markdown("### Visualization")
            st.pyplot(fig)
            st.markdown(chart_summary)
        st.stop()

    metadata_answer = answer_with_metadata(user_question)
    st.markdown(metadata_answer, unsafe_allow_html=True)
    st.stop()

    if st.session_state.get("openai_api_key") and filtered_df is not None:
        st.markdown("### 💬 AI Summary")
        llm_summary = llm_summarize_metadata(filtered_df, user_question)
        if llm_summary:
            st.write(llm_summary)
        else:
            st.info("No AI summary generated. Please verify the OpenAI key or try again.")
    else:
        st.markdown(f"**Summary:** {summary}")

if clear_btn:
    st.experimental_rerun()
if run_btn:
    if not user_question.strip():
        st.error("Please enter a question first.")
    else:
        with st.spinner("Running classification and retrieval..."):
            pred_label, pred_prob, probs_dict = classify_question(user_question)         # 1. Classifying question
            retrieved_docs = retrieve_documents(user_question, top_k=top_k)              # 2. Retrieving documents

        st.markdown("### 🧠 Category prediction")                                        # Showing classification results
        st.markdown(f"**Predicted category:** `{pred_label}`  (confidence: `{pred_prob:.2%}`)")
        prob_rows = [                                                                    # Displaying probability breakdown
            {"Category": k, "Probability": v}
            for k, v in sorted(probs_dict.items(), key=lambda x: x[1], reverse=True)
        ]
        st.table(pd.DataFrame(prob_rows))
        st.markdown("### 📚 Top retrieved RSC documents")
        if not retrieved_docs:
            st.warning("No documents retrieved. Try a different question.")
        else:
            for i, doc in enumerate(retrieved_docs, start=1):
                title = doc.get("title", f"Document {i}")
                url = doc.get("url", "")
                category = doc.get("category", "")
                year = doc.get("document_year", "")
                sim = doc.get("cosine_similarity", None)
                header = f"**[{i}] {title}**"
                if url:
                    header += f"  ·  [Link]({url})"
                st.markdown(header)
                meta = f"Category: `{category}`"
                if year:
                    meta += f" · Year: `{year}`"
                if sim is not None:
                    meta += f" · Similarity: `{sim:.3f}`"
                st.caption(meta)

        st.markdown("### 💬 RegPolicyBot answer")
        if answer_mode == "Heuristic (offline)":
            answer = generate_heuristic_answer(user_question, pred_label, retrieved_docs)
        else:
            if llm_client is None:
                st.info(
                    "OpenAI LLM not available (no key or library). "
                    "Falling back to offline heuristic answer."
                )
                answer = generate_heuristic_answer(user_question, pred_label, retrieved_docs)
            else:
                answer = generate_llm_answer(
                    llm_client,
                    user_question,
                    pred_label,
                    retrieved_docs,
                    model_name="gpt-4o-mini",
                )
        st.markdown(answer)
# ======================================================================================================================
# STREAMLIT APPLICATION SUMMARY
# ======================================================================================================================
"""
This Streamlit application completed the deployment phase of the RegPolicyBot project. It integrated the fine-tuned 
DistilBERT classifier, MiniLM retrieval index, and optional OpenAI LLM into a cohesive hybrid RAG system. The application 
distinguished between document-based queries and structured-data questions, returned metadata tables and charts when 
appropriate, and produced context-grounded answers through either heuristic summarization or LLM synthesis.

By combining classification, semantic retrieval, metadata analytics, and interactive visualization, this interface 
demonstrated how NLP techniques could support regulatory research and policy interpretation in a practical, end-to-end 
system.
"""
# ======================================================================================================================
# REFERENCES
# ======================================================================================================================

# Streamlit Documentation — UI Development, Caching, State Handling
# https://docs.streamlit.io/

# HuggingFace Transformers — DistilBERT Sequence Classification
# https://huggingface.co/docs/transformers/model_doc/distilbert

# SentenceTransformers — MiniLM Embeddings
# https://www.sbert.net/docs/pretrained_models.html

# Scikit-learn NearestNeighbors — Semantic Retrieval Backend
# https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html

# OpenAI API Documentation — Chat Completion & Response Composition
# https://platform.openai.com/docs/

# Pandas Documentation — Data Filtering and Table Formatting
# https://pandas.pydata.org/

# Matplotlib Documentation — Visualization of Metadata Trends
# https://matplotlib.org/

# These sources informed the application’s classifier loading approach, semantic retrieval design,
# metadata analytics functions, LLM prompting strategy, and Streamlit UI structure.




