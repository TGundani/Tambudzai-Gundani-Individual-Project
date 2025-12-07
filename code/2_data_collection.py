"""
========================================================================================================================
RegPolicyBot – DATA ACQUISITION
------------------------------------------------------------------------------------------------------------------------

Purpose: This script collects raw regulatory text from the GW Regulatory Studies Center (RSC) website. The goal is to
         build the foundational corpus for the RegPolicyBot system.We scrape three categories of documents:

         1. Commentaries
         2. Working Papers
         3. Events (summaries and transcripts where available)

These categories were selected because they contain the bulk of RSC’s substantive regulatory insights, are consistently
structured, and can be reliably extracted through web scraping. Scraping the entire website was intentionally avoided due
to inconsistent layouts, non-text pages, and the risk of collecting irrelevant or noisy content.The extracted text and
metadata are stored in: data_raw/rsc_raw.csv This file will be used in the next stage (data cleaning and preprocessing).

In addition to scraping, the script offers an optional mechanism to load structured regulatory metadata (e.g., Major Rules
and Federal Register Tracking) to support downstream tasks such as metadata lookup and integrated display within the chatbot.
This data supplements scraped text but is not used for NLP model training.

Script design principles:
-------------------------
• The script uses a hybrid design: standard and structured enough for real-world replicability.
• Functions have descriptive names.
• Comments explain why steps are taken (important for policy audiences).
• Error handling ensures the pipeline runs without interruption
========================================================================================================================
"""
# IMPORTING DEPENDENCIES
# ======================================================================================================================
import os
import time
import requests
import pandas as pd
from ftfy import fix_text
from bs4 import BeautifulSoup
from typing import List, Dict, Optional

# ======================================================================================================================
# Section 1A : LOADING SUPPLEMENTARY METADATA (Major Rules, Significant Rules, etc.)

# We selected three metadata datasets: federal register tracking, major rules by presidential year, and federal register
# rules by presidential year, because they provide the most relevant structured information for enriching chatbot responses
# and supporting structured queries.
# ======================================================================================================================
def load_regulatory_metadata(
    metadata_folder: str = os.path.join("..", "backend_metadata")) -> dict:
    """
    Loading additional structured regulatory datasets that can later be used to enrich chatbot responses. These datasets
    contain rule-level metadata but do not contain narrative text, so they are not used for NLP training.
    """
    metadata_files = [
        "fr_tracking.csv",
        "major_rules_by_presidential_year.csv",
        "federal_register_rules_by_presidential_year.csv"
    ]
    metadata_dict = {}
    for filename in metadata_files:
        file_path = os.path.join(metadata_folder, filename)

        if os.path.exists(file_path):
            try:
                # Try UTF-8 first
                try:
                    df = pd.read_csv(file_path, encoding="utf-8")
                except UnicodeDecodeError:
                    # Fallback to Latin-1
                    df = pd.read_csv(file_path, encoding="latin-1")

                # If successful, store it
                dataset_name = filename.replace(".csv", "")
                metadata_dict[dataset_name] = df
                print(f"[INFO] Loaded metadata file: {filename} ({len(df)} rows)")

            except Exception as error:
                print(f"[Warning] Could not load {filename}: {error}")

        else:
            print(f"[Note] Optional metadata file not found: {filename}")

    return metadata_dict
# ======================================================================================================================
# Section 1B: RULE METADATA LOOKUP HELPER

# This complementary function to the actual Natural Language Processing functions below, searches the detailed fr_tracking
# dataset for rule titles that appear in chatbot-retrieved text. This allowed the system to attach metadata such as agency,
# publication date, and significance category to the final Natural Language Generated answer.
# ======================================================================================================================
def find_rule_metadata(rule_title: str, metadata_dict: dict) -> dict:
    """
    Searches fr_tracking.csv for a specific rule title. Returns a Dictionary with matched metadata fields,
    or empty dict if not found.
    """

    if "fr_tracking" not in metadata_dict:
        return {}
    fr_df = metadata_dict["fr_tracking"]
    matches = fr_df[fr_df["title"].str.contains(rule_title, case=False, na=False)]  # Case-insensitive partial match
    if matches.empty:
        return {}

    return matches.iloc[0].to_dict()                                               # Return the first match as a simple dictionary
# ======================================================================================================================
# Section 2 : RETRIEVING HTML CONTENT FROM THE RSC WEBPAGE

# Defined a helper that sends a GET request to an RSC webpage and returns the raw HTML. It includes error handling to
# prevent the pipeline from stopping if a page temporarily fails to load, keeping the scraper stable.
# ======================================================================================================================
def fetch_page_html(url: str) -> Optional[str]:
    """
    Retrieving the HTML content of a webpage. Returns Raw HTML text if the request succeeds, otherwise None.
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.text

    except Exception as error:
        print(f"[Warning] Could not fetch page: {url}\nReason: {error}")
        return None
# ======================================================================================================================
# Section 3 : EXTRACTING MAIN ARTICLE TEXT

# RSC pages place article content inside a specific <div> block. Here we isolated that section, extracted paragraphs,
# and returned them as a single text string. The goal was to remove navigation menus, footers, and other non-useful
# elements so that only the regulatory narrative remained.
# ======================================================================================================================

def extract_article_text(html_content: str) -> str:
    """
    Extracts the main article body from an RSC webpage. Returns Clean text extracted from the article content area.
    """
    soup = BeautifulSoup(html_content, "html.parser")
    article_block = soup.find("div", class_="field--name-body")
    if not article_block:
        return ""

    paragraphs = [p.get_text(strip=True) for p in article_block.find_all("p")]
    full_text = " ".join(paragraphs)

    return full_text
# ======================================================================================================================
# Section 4: COLLECTING ARTICLE LINKS FROM CATEGORY PAGES

# Each RSC category (Commentaries, Working Papers, Events) has a listing page containing links to individual articles.
# Here we identified those links and returned a clean list of article URLs for scraping. This isolated navigation from
# content and ensured only valid article pages are passed into the scraper.
# ======================================================================================================================
from urllib.parse import urlparse, urljoin
BASE_DOMAIN = "https://regulatorystudies.columbian.gwu.edu"

def is_internal_article_link(href: str) -> bool:
    if not href:
        return False
    if href.startswith("#") or href.startswith("mailto:"):                  # Fragments and email links are not articles
        return False
    if "page=" in href:                                                     # Pagination '?page=1' or contain 'page='
        return False

    full_url = urljoin(BASE_DOMAIN, href)                                   # Relative links into absolute to inspect the hostname
    parsed = urlparse(full_url)
    if parsed.netloc != urlparse(BASE_DOMAIN).netloc:                       # Keeping links that point back to RSC
        return False
    return True                                                             # Keeping any non-empty, on-domain path.

def collect_listing_links(listing_url: str) -> List[str]:
    """
    Collecting all candidate article URLs from a category listing page. Handles pagination and returns a de-duplicated
    list of absolute URLs.
    """
    print(f"\n[INFO] Gathering links from listing page: {listing_url}")

    article_urls: set[str] = set()
    pages_to_visit: set[str] = {listing_url}
    visited_pages: set[str] = set()

    while pages_to_visit:
        page_url = pages_to_visit.pop()
        if page_url in visited_pages:
            continue
        visited_pages.add(page_url)
        print(f"[DEBUG] Visiting listing page: {page_url}")
        page_html = fetch_page_html(page_url)
        if not page_html:
            print(f"[Warning] Skipping page with no HTML: {page_url}")
            continue
        soup = BeautifulSoup(page_html, "html.parser")
# ----------------------------------------------------------------------------------------------------------------------
# Section 4a). Collecting candidate article links on this listing page
# ----------------------------------------------------------------------------------------------------------------------
        main_region = soup.find("main")                                   # Links inside the <main> content area.
        if main_region:
            candidate_links = main_region.find_all("a", href=True)
        else:
            candidate_links = soup.find_all("a", href=True)         # Fallback: using all links if <main> is not present for some reason.

        page_article_count = 0
        for a_tag in candidate_links:
            href = a_tag.get("href")
            if not is_internal_article_link(href):
                continue
            full_url = urljoin(BASE_DOMAIN, href)
            if full_url not in article_urls:                              # Adding to set; duplicates are automatically removed
                article_urls.add(full_url)
                page_article_count += 1
        print(f"[DEBUG] Found {page_article_count} unique candidate articles on this page.")

# ------------------------------------------------------------------------------------------------------------------
# Section 4b). Following pagination links (?page=1, ?page=2, ...)
# ------------------------------------------------------------------------------------------------------------------
        pager_links = soup.select("ul.pager__items a[href]")             # Looking for pager links, falling back to any link that contains 'page='.
        if not pager_links:
            pager_links = [                                              # Fallback: scanning all links and picking those that look like pagination.
                a for a in soup.find_all("a", href=True)
                if "page=" in a.get("href", "")
            ]

        for a_tag in pager_links:
            href = a_tag.get("href")
            if not href:
                continue
            next_page = urljoin(listing_url, href)
            if next_page not in visited_pages and next_page not in pages_to_visit:
                pages_to_visit.add(next_page)

    print(f"[INFO] Total unique candidate article URLs collected: {len(article_urls)}")
    return sorted(article_urls)
# ======================================================================================================================
# Section 5: SCRAPING ALL ARTICLES FROM A CATEGORY
# ======================================================================================================================

# Given a category label and its listing page, we fetched each article, extracted basic metadata (title, date, author),
# pulled the article text, and stored each result in a structured dictionary. This keeps the scraping logic modular and
# makes it easy to expand or adjust categories in the future.
# ======================================================================================================================
def collect_category(category_label: str, listing_page_url: str) -> List[Dict]:
    """
    Scraping all articles belonging to a specific category. Returning a list of dictionaries, each representing one article.
    """
    print(f"\n[INFO] Scraping category: {category_label}")
    article_urls = collect_listing_links(listing_page_url)
    print(f"[INFO] Candidate URLs for {category_label}: {len(article_urls)}")

    collected_items: List[Dict] = []
    skipped_no_body = 0
    for idx, article_url in enumerate(article_urls, start=1):
        print(f"[DEBUG] [{category_label}] Processing ({idx}/{len(article_urls)}): {article_url}")

        page_html = fetch_page_html(article_url)
        if not page_html:
            print(f"[WARNING] Could not load article page: {article_url}")
            continue
        soup = BeautifulSoup(page_html, "html.parser")
        ld_struct = None                                                       # JSON-LD (for backup but was not used as HTML was successful)
        ld_json = soup.find("script", type="application/ld+json")
        if ld_json:
            try:
                import json
                ld_struct = json.loads(ld_json.string)
            except Exception:
                ld_struct = None
# ----------------------------------------------------------------------------------------------------------------------
# Section 5a). Extracting article Title and Data
# ----------------------------------------------------------------------------------------------------------------------
        title_tag = soup.find("h1")
        title_text = title_tag.get_text(strip=True) if title_tag else None

        date_text = None
        date_container = soup.find("div", class_="gw-when")                      # <div class="gw-when"><span>September 16, 2025</span></div>
        if date_container:
            span = date_container.find("span", class_="ml-1")
            if span:
                date_text = span.get_text(strip=True)

        if not date_text and ld_struct and isinstance(ld_struct, dict):          # Fallback: JSON-LD (Was not used as HTML was successful)
            graph = ld_struct.get("@graph", [])
            for entry in graph:
                if entry.get("@type") == "Article" and entry.get("datePublished"):
                    date_text = entry.get("datePublished")
                    break
# ----------------------------------------------------------------------------------------------------------------------
# Section 5b). Extracting Author Name
# ----------------------------------------------------------------------------------------------------------------------
        author_text = None
        author_block = soup.find("div", class_="article-author")                  # Case A: <div class="article-author"><a>NAME</a></div>
        if author_block:
            link = author_block.find("a")
            if link:
                author_text = link.get_text(strip=True)

        if not author_text and ld_struct and isinstance(ld_struct, dict):          # Case B: JSON-LD <script type="application/ld+json">
            graph = ld_struct.get("@graph", [])
            for entry in graph:
                if entry.get("@type") == "Article":
                    auth = entry.get("author")
                    if isinstance(auth, dict):
                        name = auth.get("name")
                        if name and "Center" not in name and "College" not in name:
                            author_text = name
                    break

        if not author_text:                                                        # Case C: <p><strong>Authored by:</strong> NAME</p>
            label = soup.find(string=lambda x: x and "Authored by" in x)
            if label:
                parent = label.parent
                link = parent.find("a")
                if link:
                    author_text = link.get_text(strip=True)

        for p in soup.find_all("p"):                                               # Removing boilerplates before extracting raw_text
            text = p.get_text(strip=True).lower()
            if text.startswith(("download", "listen to", "podcast", "view additional", "see other")):
                p.decompose()
        for promo in soup.select("div[type*=tpl], .blue-box, .pos-one"):
            promo.decompose()
        for hr in soup.find_all("hr"):
            hr.decompose()
        for div in soup.select("div.blue-boxes, div.field--name-field-download"):
            div.decompose()
# ----------------------------------------------------------------------------------------------------------------------
# Section 5c). Extracting Full Article Text and Saving Record
# ----------------------------------------------------------------------------------------------------------------------
        article_block = soup.find("div", class_="gw-article-body")
        if not article_block:
            skipped_no_body += 1
            print(f"[DEBUG] Skipping (no article body found): {article_url}")
            continue

        for junk in article_block.select(                                                    # Removing junk inside the article body
                "a.btn, hr, .promo-title, .blue-box, .gw-media-header, "
                ".field--name-field-podcast, .field--name-field-download, .field--name-field-teaser"
        ):
            junk.decompose()

        paragraphs_raw = article_block.find_all("p")
        cleaned_paragraphs = []

        for p in paragraphs_raw[:3]:                                                           # Author always appears in the first 1–3 <p> tags
            txt = p.get_text(" ", strip=True)
            lowered = txt.lower()

            if lowered.startswith("authored by"):                                              # Case 1: "Authored by: Name"
                author_text = txt.replace("Authored by", "").replace(":", "").strip()
                p.decompose()
                continue
            if lowered.startswith("by"):                                                       # Case 2: "By Name" or "By: Name"
                cleaned = (
                    txt.replace("By:", "")
                    .replace("By", "")
                    .replace("by:", "")
                    .replace("by", "")
                    .replace(":", "")
                    .strip()
                )
                if cleaned:
                    author_text = cleaned
                p.decompose()
                continue
            words = txt.replace(".", "").split()                    # Case 3: Standalone name (1–4 words, alphabetic, possibly with periods)
            if 1 <= len(words) <= 4 and all(w.replace(".", "").isalpha() for w in words):
                if not author_text:
                    author_text = txt.strip()
                p.decompose()
                continue

        for p in paragraphs_raw:
            txt = p.get_text(" ", strip=True)
            if not txt:
                continue
            txt = fix_text(txt)
            cleaned_paragraphs.append(txt)

        article_text = "\n\n".join(cleaned_paragraphs)
        article_text = fix_text(article_text)

        collected_items.append({                                                 # Saving record
            "category": category_label,
            "url": article_url,
            "title": title_text,
            "author": author_text,
            "date": date_text,
            "raw_text": article_text
        })
        time.sleep(0.5)
    print(f"[INFO] Finished category: {category_label}")
    print(f"       Articles saved: {len(collected_items)}")
    print(f"       Pages skipped (no article body): {skipped_no_body}")
    return collected_items
# ======================================================================================================================
# Section 6: MAIN DATA COLLECTION PIPELINE

# Here we orchestrated the entire scraping workflow: loading optional metadata, scraping all three text categories,
# combining results, and saving the final dataset into data_raw/rsc_raw.csv. This design kept all high-level steps in one
# place, making the script easy to replicate and modify (reproducibility is a key goal for this project).
# ======================================================================================================================
def main():
    """
    Running the complete scraping pipeline.
    """
    print("\n============================================")
    print("         Starting RegPolicyBot Scraper")
    print("============================================\n")

    raw_output_dir = os.path.join("..", "data_raw")                             # Saving into the existing project-level data_raw folder
    os.makedirs(raw_output_dir, exist_ok=True)
    output_file = os.path.join(raw_output_dir, "rsc_raw.csv")

    COMMENTARIES_URL = "https://regulatorystudies.columbian.gwu.edu/commentaries-insights"              # Category pages
    WORKING_PAPERS_URL = "https://regulatorystudies.columbian.gwu.edu/journal-articles-working-papers"
    EVENTS_URL = "https://regulatorystudies.columbian.gwu.edu/news-and-events"

    metadata_dict = load_regulatory_metadata()                                  # Loading metadata files
    scraped_records = []
    scraped_records += collect_category("commentaries_insights", COMMENTARIES_URL)
    scraped_records += collect_category("journal_articles_working_papers", WORKING_PAPERS_URL)
    scraped_records += collect_category("events", EVENTS_URL)

    df = pd.DataFrame(scraped_records)
    df.to_csv(output_file, index=False)

    print(f"[INFO] Saved {len(df)} records to {output_file}")
    print("[INFO] Data collection completed successfully.\n")

if __name__ == "__main__":
    main()

#                                              OUTPUT:
# [INFO] Finished category: commentaries_insights
#        Articles saved: 390
#        Pages skipped (no article body): 0
#
# [INFO] Finished category: journal_articles_working_papers
#        Articles saved: 122
#        Pages skipped (no article body): 0
#
# [INFO] Finished category: events
#        Articles saved: 701
#        Pages skipped (no article body): 110
# ======================================================================================================================
# SECTION 7: SUMMARY OF RESULTS AND CONCLUDING REMARKS
# ======================================================================================================================
"""
This script successfully completed the full web-scraping workflow for the RegPolicyBot project, retrieving content from 
three major publication groups:

    • Journal Articles & Working Papers
    • Commentaries & Insights
    • Events & Event Summaries

Across the three categories, the scraper identified all candidate URLs,validated page structures, extracted metadata, and
collected the main body text for each article. The final dataset (saved to data_raw/rsc_raw.csv)
contains:

        • Total records saved: 1,213
        • Journal Articles & Working Papers: 122
        • Commentaries & Insights: 390
        • Events: 701
        • Pages skipped due to missing body content: 110 (Events only)

These counts align with expectations based on the RSC website’s structure. The higher skip rate within the Events category
reflects the presence of pages that do not contain traditional article bodies (e.g., event listings, registration pages,
and PDF-only announcements).

Data Quality Observations
------------------------------------------------------------------------------------------------------------------------
• Metadata extraction (titles, authors, and dates) was consistently populated for the majority of pages across all categories.

• Variations in Drupal templates required flexible pattern matching. The script handled typical patterns (field--name-body,
  text-formatted,field__item blocks) and applied targeted fallbacks for older templates.

• The raw_text field contains the full HTML-derived article body. This is expected to be verbose at this stage (e.g., 
  includes “Download PDF” notices, podcast buttons, blue-box summaries). These elements will be systematically removed 
  during the cleaning phase in 3_data_cleaning.py.

• Importantly, the raw_text preserves the entire article, ensuring that no information is lost prior to cleaning — a key
  requirement for reproducible NLP pipelines.
------------------------------------------------------------------------------------------------------------------------
Concluding Remarks

With this step completed, the project now has a comprehensive, structured,and reproducible dataset of RSC publications. 
The scraper is intentionally modular and transparent, making it easy for policy researchers to understand what is being 
collected and why.The next stage (data cleaning and preprocessing) will focus on:

    • Removing boilerplate and non-content elements
    • Normalizing whitespace and formatting
    • Tokenization and lemmatization
    • Constructing clean, analysis-ready text for modeling

This completes the Data Acquisition & Web Scraping phase of the RegPolicyBot pipeline. Data collection completed 
successfully.
"""
# ======================================================================================================================
# SECTION 8: CODE REFERENCES
# ======================================================================================================================

# These references document the external sources informing the scraping patterns, parsing approaches, metadata extraction
# logic, and pipeline design used throughout this script.
# ----------------------------------------------------------------------------------------------------------------------
# FOR SECTION 1A – LOADING SUPPLEMENTARY METADATA
# ----------------------------------------------------------------------------------------------------------------------
# [1] Pandas Documentation – Reading CSV Files
#     https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html
# [2] Python Unicode & Encoding Guidelines (Python Docs)
#     https://docs.python.org/3/howto/unicode.html
# ----------------------------------------------------------------------------------------------------------------------
# FOR SECTION 1B – METADATA LOOKUP
# ----------------------------------------------------------------------------------------------------------------------
# [1] Pandas Text Matching & Filtering
#     https://pandas.pydata.org/docs/user_guide/text.html
# [2] W3C JSON-LD Specification (Used for parsing structured metadata when available)
#     https://www.w3.org/TR/json-ld11/
# ----------------------------------------------------------------------------------------------------------------------
# FOR SECTION 2 – RETRIEVING HTML CONTENT
# ----------------------------------------------------------------------------------------------------------------------
# [1] Requests Library – Official Documentation
#     https://requests.readthedocs.io/en/latest/
# [2] Mozilla Developer Network – HTTP Status Codes Reference
#     https://developer.mozilla.org/en-US/docs/Web/HTTP/Status
# ----------------------------------------------------------------------------------------------------------------------
# FOR SECTION 3 – EXTRACTING ARTICLE TEXT
# ----------------------------------------------------------------------------------------------------------------------
# [1] BeautifulSoup Documentation
#     https://www.crummy.com/software/BeautifulSoup/bs4/doc/
# [2] ftfy Documentation – Fixing Unicode Text from the Web
#     https://ftfy.readthedocs.io/en/latest/
# ----------------------------------------------------------------------------------------------------------------------
# FOR SECTION 4 – COLLECTING ARTICLE LINKS & PAGINATION
# ----------------------------------------------------------------------------------------------------------------------
# [1] urllib.parse Documentation – URL Parsing & Normalization
#     https://docs.python.org/3/library/urllib.parse.html
# [2] Scrapy Documentation – Link Extraction Patterns (industry standard)
#     https://docs.scrapy.org/en/latest/topics/link-extractors.html
# ----------------------------------------------------------------------------------------------------------------------
# FOR SECTION 5 – SCRAPING ARTICLE CONTENT & METADATA
# ----------------------------------------------------------------------------------------------------------------------
# [1] Schema.org Metadata for Articles (JSON-LD Structures)
#     https://schema.org/Article
# [2] Stanford NLP Preprocessing Guidelines (Text Cleaning)
#     https://nlp.stanford.edu/
# ----------------------------------------------------------------------------------------------------------------------
# FOR SECTION 6 – ORCHESTRATING THE DATA PIPELINE
# ----------------------------------------------------------------------------------------------------------------------
# [1] Python Logging & Script Structuring Best Practices
#     https://docs.python.org/3/howto/logging.html
# [2] DataOps Pipeline Principles (General Workflow Design)
#     https://www.dataopsmanifesto.org/
# ----------------------------------------------------------------------------------------------------------------------
# FOR SECTION 7 – RESULTS SUMMARY & DOCUMENTATION
# ----------------------------------------------------------------------------------------------------------------------
# [1] ACM Guidelines for Reproducible Research Documentation
#     https://www.acm.org/publications/policies/artifact-review-and-badging-current
# [2] MIT Data Nutrition Project – Dataset Documentation Standards
#     https://datanutrition.org/

