# term_statistics.py

import numpy as np
import pandas as pd
from collections import Counter
from typing import Dict

from tfidf_model import TfIdfModel


class TermStatistics:
    """
    Computes descriptive statistics over a TF-IDF corpus and vocabulary.
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
            raise RuntimeError("TfIdfModel must be fitted before computing statistics")

    # --------------------------------------------------
    # Corpus-level statistics
    # --------------------------------------------------

    def corpus_stats(self) -> Dict[str, float]:
        """
        Statistics about document length (in words).
        """
        df = self.tfidf.df
        text_col = self.tfidf.text_column

        n_words = df[text_col].fillna("").astype(str).str.split().str.len()

        return {
            "n_documents": len(df),
            "mean_words": n_words.mean(),
            "median_words": n_words.median(),
            "std_words": n_words.std(),
            "min_words": n_words.min(),
            "max_words": n_words.max(),
        }

    # --------------------------------------------------
    # Lexical statistics
    # --------------------------------------------------

    def lexical_stats(self) -> Dict[str, float]:
        """
        Vocabulary-level statistics.
        """
        df = self.tfidf.df
        text_col = self.tfidf.text_column

        all_tokens = " ".join(
            df[text_col].fillna("").astype(str).tolist()
        ).split()

        token_counts = Counter(all_tokens)

        vocab_size = len(token_counts)
        total_tokens = sum(token_counts.values())

        return {
            "total_tokens": total_tokens,
            "vocabulary_size": vocab_size,
            "type_token_ratio": vocab_size / total_tokens if total_tokens > 0 else 0.0,
        }

    # --------------------------------------------------
    # TF-IDF matrix statistics
    # --------------------------------------------------

    def tfidf_matrix_stats(self) -> Dict[str, float]:
        """
        Statistics of the TF-IDF matrix.
        """
        X = self.tfidf.get_matrix()
        n_docs, n_terms = X.shape
        density = X.nnz / (n_docs * n_terms)

        return {
            "n_documents": n_docs,
            "n_terms": n_terms,
            "matrix_density": density,
            "matrix_sparsity": 1.0 - density,
        }

    # --------------------------------------------------
    # Term relevance
    # --------------------------------------------------

    def mean_tfidf_terms(self) -> pd.DataFrame:
        """
        Mean TF-IDF per term.
        """
        return self.tfidf.term_stats()

