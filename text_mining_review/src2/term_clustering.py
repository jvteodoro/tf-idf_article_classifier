# term_clustering.py

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from typing import Optional, List

from tfidf_model import TfIdfModel


class TermClustering:
    """
    Performs semantic clustering of TF-IDF terms.
    """

    def __init__(
        self,
        tfidf_model: TfIdfModel
    ):
        """
        Parameters
        ----------
        tfidf_model : TfIdfModel
            A fitted TF-IDF model.
        """
        self.tfidf = tfidf_model

        if self.tfidf.X_tfidf is None:
            raise RuntimeError("TfIdfModel must be fitted before clustering")

    # --------------------------------------------------
    # Public API
    # --------------------------------------------------

    def cluster_terms(
        self,
        term_df: pd.DataFrame,
        n_clusters: int,
        linkage: str = "ward"
    ) -> pd.DataFrame:
        """
        Cluster a subset of terms using hierarchical clustering.

        Parameters
        ----------
        term_df : DataFrame
            Must contain columns ['term', 'mean_tfidf'].
        n_clusters : int
            Number of clusters to generate.
        linkage : str
            Linkage method (default: 'ward').

        Returns
        -------
        DataFrame
            term | mean_tfidf | cluster
        """
        self._validate_term_df(term_df)

        # Map terms to column indices in TF-IDF matrix
        vocab = self.tfidf.vectorizer.vocabulary_

        term_indices = [
            vocab[t] for t in term_df["term"]
            if t in vocab
        ]

        if len(term_indices) < n_clusters:
            raise ValueError(
                "Number of terms must be >= number of clusters"
            )

        # Build term-document matrix
        X_terms = self.tfidf.X_tfidf[:, term_indices].T.toarray()

        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage,
            metric="euclidean"
        )

        labels = model.fit_predict(X_terms)

        out = term_df.copy()
        out = out.iloc[:len(labels)].copy()
        out["cluster"] = labels

        return out.sort_values(
            ["cluster", "mean_tfidf"],
            ascending=[True, False]
        ).reset_index(drop=True)

    # --------------------------------------------------
    # Helpers
    # --------------------------------------------------

    def _validate_term_df(self, term_df: pd.DataFrame) -> None:
        required = {"term", "mean_tfidf"}
        missing = required - set(term_df.columns)

        if missing:
            raise KeyError(
                f"term_df is missing required columns: {missing}"
            )

