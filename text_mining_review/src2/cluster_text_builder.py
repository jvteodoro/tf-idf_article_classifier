# cluster_text_builder.py

import pandas as pd
from typing import List, Optional


class ClusterTextBuilder:
    """
    Utility class to generate base texts for research questions
    from TF-IDF term statistics and (optionally) semantic clusters.

    This class is intentionally general: some strategies may not be
    used in a given systematic review if the corpus does not exhibit
    meaningful thematic separation.
    """

    # --------------------------------------------------
    # Initialization
    # --------------------------------------------------

    def __init__(
        self,
        term_df: pd.DataFrame,
        cluster_df: Optional[pd.DataFrame] = None
    ):
        """
        Parameters
        ----------
        term_df : DataFrame
            Must contain at least ['term', 'mean_tfidf'].
        cluster_df : DataFrame, optional
            Must contain ['term', 'cluster', 'mean_tfidf'].
            Required only for cluster-based strategies.
        """
        if not {"term", "mean_tfidf"}.issubset(term_df.columns):
            raise KeyError("term_df must contain 'term' and 'mean_tfidf'")

        self.term_df = term_df.copy()

        if cluster_df is not None:
            if not {"term", "cluster", "mean_tfidf"}.issubset(cluster_df.columns):
                raise KeyError(
                    "cluster_df must contain 'term', 'cluster', 'mean_tfidf'"
                )
            self.cluster_df = cluster_df.copy()
        else:
            self.cluster_df = None

    # --------------------------------------------------
    # Strategy 1 — Global TF-IDF (DEFAULT / BASELINE)
    # --------------------------------------------------

    def from_top_terms(
        self,
        top_k_terms: int = 20
    ) -> str:
        """
        Generate a base text from the globally most relevant terms
        ranked by mean TF-IDF.

        This is the preferred strategy when the corpus exhibits a
        cohesive vocabulary without clear thematic subclusters.

        Returns
        -------
        str
            Base text.
        """
        terms = (
            self.term_df
            .sort_values("mean_tfidf", ascending=False)
            .head(top_k_terms)["term"]
            .tolist()
        )

        return " ".join(terms)

    # --------------------------------------------------
    # Strategy 2 — Cluster-based aggregation
    # --------------------------------------------------

    def from_top_clusters(
        self,
        top_clusters: pd.DataFrame,
        top_k_clusters: int = 3,
        top_k_terms: int = 15
    ) -> pd.DataFrame:
        """
        Generate base texts from the most relevant clusters.

        Parameters
        ----------
        top_clusters : DataFrame
            Must contain ['cluster', 'similarity'].
        top_k_clusters : int
            Number of clusters to consider.
        top_k_terms : int
            Number of terms per cluster.

        Returns
        -------
        DataFrame with columns:
        - cluster
        - similarity
        - base_text
        """
        if self.cluster_df is None:
            raise RuntimeError(
                "cluster_df was not provided; cluster-based strategy unavailable"
            )

        if not {"cluster", "similarity"}.issubset(top_clusters.columns):
            raise KeyError("top_clusters must contain 'cluster' and 'similarity'")

        selected = top_clusters.head(top_k_clusters)

        rows = []

        for _, row in selected.iterrows():
            cid = row["cluster"]
            similarity = row["similarity"]

            terms = (
                self.cluster_df[self.cluster_df["cluster"] == cid]
                .sort_values("mean_tfidf", ascending=False)
                .head(top_k_terms)["term"]
                .tolist()
            )

            rows.append({
                "cluster": cid,
                "similarity": similarity,
                "base_text": " ".join(terms),
                "strategy": "cluster_top_terms"
            })

        return pd.DataFrame(rows)

    # --------------------------------------------------
    # Strategy 3 — Weighted cluster-based aggregation
    # --------------------------------------------------

    def from_weighted_clusters(
        self,
        top_clusters: pd.DataFrame,
        top_k_clusters: int = 3,
        top_k_terms: int = 15,
        scale: int = 10
    ) -> pd.DataFrame:
        """
        Generate cluster-based texts with term repetition weighted
        by cluster similarity.

        This strategy amplifies clusters more aligned with the RQ.

        Returns
        -------
        DataFrame with columns:
        - cluster
        - similarity
        - base_text
        """
        if self.cluster_df is None:
            raise RuntimeError("cluster_df required for weighted strategy")

        rows = []

        for _, row in top_clusters.head(top_k_clusters).iterrows():
            cid = row["cluster"]
            similarity = row["similarity"]

            repetition = max(1, int(similarity * scale))

            terms = (
                self.cluster_df[self.cluster_df["cluster"] == cid]
                .sort_values("mean_tfidf", ascending=False)
                .head(top_k_terms)["term"]
                .tolist()
            )

            weighted_terms = []
            for term in terms:
                weighted_terms.extend([term] * repetition)

            rows.append({
                "cluster": cid,
                "similarity": similarity,
                "base_text": " ".join(weighted_terms),
                "strategy": "cluster_weighted"
            })

        return pd.DataFrame(rows)

    # --------------------------------------------------
    # Strategy 4 — Hybrid (manual + automatic)
    # --------------------------------------------------

    def hybrid_text(
        self,
        manual_text: str,
        auto_text: str,
        alpha: float = 0.6
    ) -> str:
        """
        Combine manual and automatic base texts.

        Parameters
        ----------
        manual_text : str
            Researcher-defined text.
        auto_text : str
            Automatically generated text.
        alpha : float
            Weight for manual text (0–1).

        Returns
        -------
        str
            Hybrid base text.
        """
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be between 0 and 1")

        manual_tokens = manual_text.split()
        auto_tokens = auto_text.split()

        n_manual = int(alpha * len(manual_tokens))
        n_auto = max(1, int((1 - alpha) * len(auto_tokens)))

        hybrid_tokens = manual_tokens[:n_manual] + auto_tokens[:n_auto]

        return " ".join(hybrid_tokens)

