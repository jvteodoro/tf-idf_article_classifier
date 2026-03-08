# similarity_ranker.py

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Optional, Literal

from tfidf_model import TfIdfModel


class SimilarityRanker:
    """
    Ranks documents, terms, or clusters by semantic similarity
    to a reference string using TF-IDF and cosine similarity.
    """

    def __init__(self, tfidf_model: TfIdfModel):
        """
        Parameters
        ----------
        tfidf_model : TfIdfModel
            A fitted TF-IDF model.
        """
        self.tfidf = tfidf_model

        if self.tfidf.X_tfidf is None:
            raise RuntimeError("TfIdfModel must be fitted before ranking")

    # --------------------------------------------------
    # Core utilities
    # --------------------------------------------------

    def _vectorize_query(self, query: str):
        return self.tfidf.vectorizer.transform([query])

    # --------------------------------------------------
    # Document ranking
    # --------------------------------------------------

    def rank_documents(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Rank documents by similarity to a query string.
        """
        q_vec = self._vectorize_query(query)
        sims = cosine_similarity(q_vec, self.tfidf.X_tfidf).ravel()

        order = np.argsort(sims)[::-1]
        if top_k:
            order = order[:top_k]

        df_out = self.tfidf.df.iloc[order].copy()
        df_out["similarity"] = sims[order]

        return df_out.reset_index(drop=True).drop_duplicates(subset=["doi"])

    # --------------------------------------------------
    # Term ranking
    # --------------------------------------------------

    def rank_terms(
        self,
        query: str,
        term_df: pd.DataFrame,
        top_k: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Rank terms by similarity to a query string.
        """
        vocab = self.tfidf.vectorizer.vocabulary_

        term_indices = [
            vocab[t] for t in term_df["term"]
            if t in vocab
        ]

        X_terms = self.tfidf.X_tfidf[:, term_indices].T
        q_vec = self._vectorize_query(query)

        sims = cosine_similarity(q_vec, X_terms).ravel()
        order = np.argsort(sims)[::-1]

        if top_k:
            order = order[:top_k]

        out = term_df.iloc[order].copy()
        out["similarity"] = sims[order]

        return out.reset_index(drop=True)

    # --------------------------------------------------
    # Cluster ranking
    # --------------------------------------------------
    def rank_clusters(
    self,
    query: str,
    clustered_terms: pd.DataFrame,
    top_k: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Rank clusters by similarity to a query string
        using centroids in the TERM space (TF-IDF).
        """
        required = {"term", "cluster"}
        if not required.issubset(clustered_terms.columns):
            raise KeyError("clustered_terms must contain 'term' and 'cluster'")

        vocab = self.tfidf.vectorizer.vocabulary_
        n_terms = len(vocab)

        cluster_vectors = []
        cluster_ids = []

        # Precompute mean TF-IDF per term (1 x n_terms)
        term_means = np.asarray(self.tfidf.X_tfidf.mean(axis=0)).ravel()

        for cid, group in clustered_terms.groupby("cluster"):
            vec = np.zeros(n_terms)

            for term in group["term"]:
                if term in vocab:
                    idx = vocab[term]
                    vec[idx] = term_means[idx]

            # Normalize cluster vector (important for cosine similarity)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm

            cluster_vectors.append(vec)
            cluster_ids.append(cid)

        X_clusters = np.vstack(cluster_vectors)

        # Vectorize query (same term space!)
        q_vec = self._vectorize_query(query).toarray()

        sims = cosine_similarity(q_vec, X_clusters).ravel()
        order = np.argsort(sims)[::-1]

        if top_k:
            order = order[:top_k]

        return pd.DataFrame({
            "cluster": np.array(cluster_ids)[order],
            "similarity": sims[order]
        }).reset_index(drop=True)

