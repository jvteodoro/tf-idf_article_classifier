# tf-idf_article_classifier
# Problem Statement

Data centers have experienced a continuous increase in energy consumption, which leads to rising operational costs and broader environmental and social impacts. At the same time, modern data center infrastructures require increasingly complex architectural decisions involving trade-offs between computational performance, service quality, and operational cost.

Digital Twins have emerged as a promising technology for addressing these challenges. A Digital Twin provides a digital representation of a physical system capable of real-time monitoring, simulation, and decision support. By integrating operational data with analytical models, Digital Twins may enable the evaluation of architectural design decisions and their impact on system performance and cost efficiency.

Given this context, the central research problem addressed in this study is:

> **Can Digital Twins be used to support architectural decision evaluation in data centers by analyzing the relationship between computational performance, service delivery, and operational cost?**

To investigate this problem, the study addresses the following research questions.

## Research Questions

**RQ1**  
How are Digital Twins currently being applied in data center environments?

**RQ2**  
How are Digital Twins used to evaluate the impact of system architectural design decisions?

**RQ3**  
How can Digital Twins be applied in data centers to evaluate the impact of architectural decisions on computational performance and service cost?

---

# Methodology

To answer the proposed research questions, this study adopts a **systematic literature review supported by text mining techniques**. The methodology combines bibliographic database analysis, corpus construction, and semantic ranking of scientific publications.

The overall process consists of four main stages:

1. **Scientific database identification**
    
2. **Systematic literature search**
    
3. **Corpus construction and normalization**
    
4. **Text mining–based ranking of relevant articles**
    

---

# Scientific Database Identification

Before conducting the systematic search, a **database survey** was performed to identify the most relevant scientific databases used in Digital Twin research.

The survey was conducted using the Web of Science database as a starting point. The following query was used:

digital twin AND systematic review

The search was limited to **review papers and highly cited articles**.

The objective of this step was to identify which databases are most frequently used in systematic reviews on Digital Twins. Articles that did not explicitly describe the databases used in their methodology were excluded.

The retrieved systematic reviews were analyzed to extract the scientific databases used in their search strategies. The frequencies of database mentions were then used to estimate **Bradford zones** for Digital Twin research databases.

The analysis showed that the most frequently used databases are:

- Scopus
    
- ScienceDirect
    
- IEEE Xplore
    

To validate this ranking, the same query was executed in **Scopus**, allowing the process to be iteratively bootstrapped. This procedure ensures that the selected databases represent the **core literature sources for Digital Twin research**.

Using Bradford's law, databases were then classified into **core (Zone 1) and peripheral (Zone 2) sources**.

The results indicate that **80.89% of database mentions originate from 28% of the identified databases**, which were therefore classified as the core sources for this study.

---

# Systematic Literature Search

To retrieve relevant publications, search queries were designed based on the research questions.

All queries were limited to:

TITLE-ABS-KEY

in order to improve precision and ensure conceptual relevance.

The following core concepts were used:

- Digital Twin
    
- Data Center
    
- Performance
    
- Architecture
    
- Cost
    

The final query used across the databases was:

"digital twin"  
AND ("data center" OR "data centre")  
AND (performance OR architecture OR architectural OR cost OR financial)

The search was conducted in the following databases:

- Scopus
    
- Web of Science
    
- IEEE Xplore
    
- ScienceDirect
    
- ACM Digital Library
    

Each database export was stored separately and later merged during the data unification stage.

---

# Inclusion and Exclusion Criteria

## Inclusion Criteria

Studies were included if they satisfied at least one of the following criteria:

- **I1** — Study describes or applies Digital Twins in a data center environment
    
- **I2** — Study evaluates architectural design decisions using Digital Twins
    
- **I3** — Study reports performance or financial metrics related to system evaluation
    
- **I4** — Peer-reviewed publication
    
- **I5** — Full text available
    

## Exclusion Criteria

Studies were excluded if they satisfied any of the following conditions:

- **E1** — Study does not focus on Digital Twins
    
- **E2** — Digital Twin application is in a domain unrelated to computing infrastructure
    
- **E3** — Non peer-reviewed publication
    
- **E4** — Duplicate record
    
- **E5** — Full text not accessible
    
- **E6** — Non-English publication
    

---

# Data Acquisition

The search results from each database were exported using the native export tools.

Examples of queries executed:

### Scopus

TITLE-ABS-KEY("digital twin")  
AND TITLE-ABS-KEY("data center" OR "data centre")  
AND TITLE-ABS-KEY(performance OR architecture OR architectural OR cost OR financial)

### Web of Science

Topic: "digital twin"  
AND ("data center" OR "data centre")  
AND (performance OR architecture OR architectural OR cost OR financial)

### IEEE Xplore

("All Metadata": digital twin)  
AND ("All Metadata": data center OR "All Metadata": data centre)  
AND ("All Metadata": performance OR architecture OR architectural OR cost OR financial)

The exported datasets were stored in their original formats (CSV, RIS, XLSX, or BibTeX) for later processing.

---

# Data Unification and Corpus Construction

Because each database uses a different export schema, a **canonical metadata structure** was defined to unify all records.

The canonical schema includes:

title  
authors  
year  
source_title  
document_type  
abstract  
keywords  
doi  
publisher  
volume  
issue  
start_page  
end_page  
cited_by  
database_source  
record_id

Each database export was mapped to this schema through a **column translation table**.

To construct the textual corpus used for text mining, a new field was created by concatenating:

title + abstract + keywords

This field represents the **semantic representation of each document**.

The final corpus contained **1,724 documents**.

---

# Text Vectorization

The corpus was transformed into a vector space representation using **TF-IDF (Term Frequency–Inverse Document Frequency)**.

TF-IDF was selected because it:

- is interpretable
    
- ensures methodological reproducibility
    
- reduces the influence of extremely common terms
    

The vectorizer used **n-grams ranging from 1 to 2 words**.

Corpus statistics:

- Documents: **1,724**
    
- Vocabulary size: **9,710 terms**
    
- Average document length: **242 words**
    

The TF-IDF matrix presented a sparsity of **98.6%**, which is typical for large scientific text corpora.

---

# Document Ranking using Cosine Similarity

To identify the most relevant articles for each research question, a **semantic ranking approach** was adopted.

Each research question was represented by a **synthetic reference document**, composed of:

- title
    
- abstract
    
- keywords
    

These synthetic documents were designed to mimic the statistical structure of scientific publications.

Each reference document was then vectorized using the **same TF-IDF pipeline** applied to the corpus.

Relevance between documents and research questions was computed using **cosine similarity**:

$sim(di,q)=di⋅q∣∣di∣∣ ∣∣q∣∣sim(d_i, q) = \frac{d_i \cdot q}{||d_i|| \, ||q||}sim(di​,q)=∣∣di​∣∣∣∣q∣∣di​⋅q​$

where:

- did_idi​ is the TF-IDF vector of document iii
    
- qqq is the TF-IDF vector of the research question representation
    

All documents were ranked by similarity, and the **top 50 articles for each research question** were selected for detailed analysis.

---

# Research Question Representation

To reduce researcher bias while preserving conceptual intent, dominant terms were extracted from the corpus using **mean TF-IDF scores computed over unigrams, bigrams, and trigrams**.

These terms were used to expand the manually defined research question representations.

For example, the base representation for **RQ1** was defined as:

Title  
Digital Twins Applications in Data Centers

Abstract  
This study explores how Digital Twins can be applied in data center environments to monitor, analyze, and support decision-making related to infrastructure subsystems such as cooling systems, energy management, IT infrastructure, and network performance.

Keywords  
digital twin, data center, infrastructure monitoring, resource management, observability, decision support

---

# Research Question Analysis

The research questions are addressed as follows:

### RQ1

Literature describing Digital Twin applications in data center environments will be analyzed to identify existing use cases, architectures, and operational models.

### RQ2

Studies focusing on **architectural decision evaluation** using Digital Twins will be analyzed to understand how simulation, monitoring, and modeling techniques support system design decisions.

### RQ3

The third research question will be addressed by **synthesizing the knowledge obtained from RQ1 and RQ2**, and applying these insights to the design of a **Digital Twin prototype capable of evaluating performance-cost trade-offs in data center architectures**.
