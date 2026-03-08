"""
==========================================================
SYSTEMATIC REVIEW PIPELINE — DIGITAL TWINS IN DATA CENTERS
==========================================================

This script implements the complete methodological pipeline
used in the systematic review addressing the following RQs:

RQ1: How are digital twins being used in data center environments?
RQ2: How are digital twins used to evaluate systems architectural
     design decisions?
RQ3: How can digital twins be applied in data centers to evaluate
     the impact of architectural decisions on performance and cost?

The pipeline combines:
- traditional systematic review practices (PRISMA-oriented)
- semantic text mining (TF-IDF + cosine similarity)
- exploratory diagnostics (lexical statistics, clustering)

The goal is NOT to replace manual analysis, but to PRIORITIZE
relevant literature in a reproducible and justifiable way.
"""

from pathlib import Path

# ==========================================================
# Import core modules of the pipeline
# ==========================================================

# --- Data acquisition and normalization ---
from import_handler import ImportHandler
from normalizer import Normalizer

# --- Corpus construction and representation ---
from corpus_builder import CorpusBuilder
from tfidf_model import TfIdfModel
from term_statistics import TermStatistics

# --- Exploratory semantic diagnostics ---
from dendrogram_analyzer import DendrogramAnalyzer
from term_clustering import TermClustering

# --- Semantic ranking ---
from similarity_ranker import SimilarityRanker
import pandas as pd



# ==========================================================
# 1. CONFIGURATION
# ==========================================================
#
# This section defines:
# - where the raw data is located
# - how each database maps its metadata to a canonical schema
#
# This explicit configuration is crucial for:
# - reproducibility
# - transparency
# - auditability of the review process
#

DATA_DIR = Path("../input_data")

FILES = {
    "scopus": DATA_DIR / "scopus_export_Dec 23-2025_54e9fea9-99a0-4c64-b0e3-04934aa9b458.csv",
    "wos": DATA_DIR / "2025.12.23.wos.xls",
    "ieee": DATA_DIR / "export2025.12.23-09.07.24.csv",
    "sd": DATA_DIR / "ScienceDirect_citations_1766499024897.ris",
    "acm_abs": DATA_DIR / "acm.bib",
    "acm_kw": DATA_DIR / "acm_3679240.3734677.bib",
}

# Canonical mapping allows heterogeneous databases
# to be merged into a single, unified corpus.
COLUMN_MAPS = {
    "scopus": {
        "title": "Title",
        "abstract": "Abstract",
        "keywords": ["Author Keywords", "Index Keywords"],
        "authors": "Authors",
        "year": "Year",
        "doi": "DOI",
        "source_title": "Source title",
        "publisher": "Publisher",
        "document_type": "Document Type",
        "record_id": "EID",
    },
    "wos": {
        "title": "Article Title",
        "abstract": "Abstract",
        "keywords": ["Author Keywords", "Keywords Plus"],
        "authors": "Authors",
        "year": "Publication Year",
        "doi": "DOI",
        "source_title": "Source Title",
        "publisher": "Publisher",
        "document_type": "Document Type",
        "record_id": "UT (Unique WOS ID)",
    },
    "ieee": {
        "title": "Document Title",
        "abstract": "Abstract",
        "keywords": ["Author Keywords", "IEEE Terms", "Mesh_Terms"],
        "authors": "Authors",
        "year": "Publication Year",
        "doi": "DOI",
        "source_title": "Publication Title",
        "publisher": "Publisher",
        "document_type": None,
        "record_id": "Document Identifier",
    },
    "sd": {
        "title": "primary_title",
        "abstract": "abstract",
        "keywords": "keywords",
        "authors": "authors",
        "year": "year",
        "doi": "doi",
        "source_title": "journal_name",
        "publisher": "publisher",
        "document_type": "type_of_reference",
        "record_id": "urls",
    },
    "acm": {
        "title": "title",
        "abstract": "abstract",
        "keywords": "keywords",
        "authors": "author",
        "year": "year",
        "doi": "doi",
        "source_title": "booktitle",
        "publisher": "publisher",
        "document_type": "ENTRYTYPE",
        "record_id": "ID",
    },
}

# Initial manual formulation of RQ1.
# This will later be contrasted with data-driven base texts.
# RQ1_TEXT = (
#     "digital twins applied in data center environments"
# )
# RQ1_TEXT = (
#     "digital twins applied in data center environments"
# )
# RQ2_TEXT = (
#     "digital twins used to evaluate  the impact of architecture decisions"
# )

# RQ3_TEXT = (
#     "digital twins used to monitor and evaluate performance-related and cost decision making"
# )

# Iteration 2
# RQ1_TEXT = (
#     "digital twins applied in data center environments "
#     "for modeling monitoring simulation and system analysis"
# )
# RQ2_TEXT = (
#     "digital twins applied in data center computing infrastructure "
#     "to evaluate alternative architecture designs and system-level decisions"
# )
# RQ3_TEXT = (
#     "digital twins applied in data center computing infrastructure "
#     "to evaluate computing performance energy efficiency and operational cost"
# )

# Iteration 3
RQ1_TEXT = (
    " data center modeling simulation monitoring control integration "
    "operational management workload orchestration fault prediction "
    "resource management system optimization"
)

RQ2_TEXT = (
    "data center architectural design decisions topology configuration "
    "resource allocation scheduling virtualization hardware software co-design "
    "scalability resilience redundancy system architecture trade-offs"
)

RQ3_TEXT = (
    " data center performance latency throughput quality of service SLA "
    "energy efficiency power consumption thermal management "
    "operational cost total cost of ownership OPEX CAPEX "
    "service reliability availability"
)




# ==========================================================
# 2. DATA ACQUISITION
# ==========================================================
#
# All databases are imported exactly as exported.
# No filtering is applied at this stage besides what
# was already done via search queries.
#
pd.set_option('display.max_colwidth', None)

importer = ImportHandler(FILES)
raw_dfs = importer.load_all()

print("Databases successfully imported:")
for k, v in raw_dfs.items():
    print(f" - {k}: {len(v)} records")

# ==========================================================
# 3. NORMALIZATION AND UNIFICATION
# ==========================================================
#
# This step converts heterogeneous metadata schemas
# into a unified, canonical representation.
#
# This is a prerequisite for any cross-database analysis.
#

normalizer = Normalizer(COLUMN_MAPS)
normalized = normalizer.normalize_all(raw_dfs)
df_all = normalizer.unify(normalized)

print("\nAfter normalization and unification:")
print(f"Total unique records in corpus: {len(df_all)}")

# ==========================================================
# 4. CORPUS CONSTRUCTION
# ==========================================================
#
# A single textual field is built per document by
# concatenating title, abstract, and keywords.
#
# This reflects standard practice in bibliometric
# and text-mining-based systematic reviews.
#

corpus_builder = CorpusBuilder()
df_text = corpus_builder.build(df_all)

# ==========================================================
# 5. TF-IDF REPRESENTATION
# ==========================================================
#
# TF-IDF is used to:
# - emphasize discriminative terms
# - reduce the impact of generic vocabulary
# - enable vector-space similarity comparisons
#

tfidf = TfIdfModel(
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95
)
tfidf.fit(df_text)

# ==========================================================
# 6. DESCRIPTIVE STATISTICS (JUSTIFICATION STEP)
# ==========================================================
#
# These statistics are not decorative.
# They justify:
# - choice of representation
# - need for automated prioritization
# - limitations of clustering
#

stats = TermStatistics(tfidf)

print("\n================ CORPUS STATISTICS ================")
print(stats.corpus_stats())

print("\n================ LEXICAL STATISTICS ===============")
print(stats.lexical_stats())

print("\n================ TF-IDF MATRIX ====================")
print(stats.tfidf_matrix_stats())

term_df = stats.mean_tfidf_terms()

# ==========================================================
# 7. CORE VOCABULARY EXTRACTION
# ==========================================================
#
# We extract the most relevant terms (top 5%)
# to study whether the corpus exhibits multiple
# semantic subdomains or a single cohesive vocabulary.
#

threshold = term_df["mean_tfidf"].quantile(0.95)
core_terms = term_df[term_df["mean_tfidf"] >= threshold]

print("\nCore vocabulary analysis:")
print(f" - Total terms: {len(term_df)}")
print(f" - Core terms (top 5%): {len(core_terms)}")

# ==========================================================
# 8. CLUSTERING AS A DIAGNOSTIC TOOL (NOT THEMATIC SEGMENTATION)
# ==========================================================
#
# Hierarchical clustering is applied ONLY to investigate
# whether the corpus naturally splits into distinct themes.
#
# The result will guide — not dictate — methodological choices.
#

X_terms = tfidf.get_matrix()[:, [
    tfidf.vectorizer.vocabulary_[t]
    for t in core_terms["term"]
]].T

# dendro = DendrogramAnalyzer(X_terms)
# dendro.plot()

# clusterer = TermClustering(tfidf)
# cluster_df = clusterer.cluster_terms(
#     term_df=core_terms,
#     n_clusters=10
# )

# print("\nCluster distribution (diagnostic):")
# print(cluster_df["cluster"].value_counts())

# ==========================================================
# 9. SEMANTIC RANKING OF DOCUMENTS (PRIMARY FILTERING MECHANISM)
# ==========================================================
#
# Since clustering does not reveal clear thematic separation,
# we prioritize documents directly via semantic similarity
# between RQ base text and document vectors.
#

ranker = SimilarityRanker(tfidf)

top_docs_rq1 = ranker.rank_documents(
    query=RQ1_TEXT
)
top_docs_rq2 = ranker.rank_documents(
    query=RQ2_TEXT
)
top_docs_rq3 = ranker.rank_documents(
    query=RQ3_TEXT
)
print(f"\nRQ1: {RQ1_TEXT}")
print(f"\nFound {top_docs_rq1.shape[0]}. Top-ranked documents for RQ1:")
print(top_docs_rq1[["title", "year", "doi", "similarity"]].head(10))

print(f"\nFound {top_docs_rq2.shape[0]} RQ2: {RQ2_TEXT}")
print("\nTop-ranked documents for RQ2:")
print(top_docs_rq2[["title", "year", "doi", "similarity"]].head(10))

print(f"\nFound {top_docs_rq3.shape[0]} RQ3: {RQ3_TEXT}")
print("\nTop-ranked documents for RQ3:")
print(top_docs_rq3[["title", "year", "doi", "similarity"]].head(10))


intersection_12 = pd.merge(top_docs_rq1, top_docs_rq2, on="doi", how="inner")

intersection_13 = pd.merge(top_docs_rq1, top_docs_rq3, on="doi", how="inner")

print(f"Found {intersection_12.shape[0]} intersection between RQ1 and RQ2")
print(intersection_12.head(10))

print("\nNOTE:")
print(
    "Document ranking, rather than term clustering, "
    "is adopted as the primary mechanism for evidence "
    "prioritization due to the cohesive semantic structure "
    "of the corpus."
)
with pd.ExcelWriter("../pipeline_output.xlsx", engine='openpyxl') as writer:
    top_docs_rq1.to_excel(writer, sheet_name="RQ1")
    top_docs_rq2.to_excel(writer, sheet_name="RQ2")
    top_docs_rq3.to_excel(writer, sheet_name="RQ3")


