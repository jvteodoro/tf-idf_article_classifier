# tfidf_model.py

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Optional, Dict


class TfIdfModel:
    """
    Builds and manages a TF-IDF representation of a textual corpus.
    """

    def __init__(
        self,
        text_column: str = "text",
        ngram_range: tuple = (1, 2),
        min_df: int | float = 2,
        max_df: int | float = 0.95,
        stop_words: Optional[str] = "english",
        max_features: Optional[int] = None,
    ):
        """
        Parameters
        ----------
        text_column : str
            Name of the column containing the corpus text.
        ngram_range : tuple
            n-gram range for TF-IDF.
        min_df : int or float
            Minimum document frequency.
        max_df : int or float
            Maximum document frequency.
        stop_words : str or None
            Stopwords configuration.
        max_features : int or None
            Maximum number of features.
        """
        self.text_column = text_column

        self.vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            stop_words=stop_words,
            max_features=max_features,
            norm="l2"
        )

        self.X_tfidf = None
        self.feature_names = None
        self.df = None

    # --------------------------------------------------
    # Public API
    # --------------------------------------------------

    def fit(self, df: pd.DataFrame) -> None:
        """
        Fit the TF-IDF model on the corpus.

        Parameters
        ----------
        df : DataFrame
            DataFrame containing the text column.
        """
        if self.text_column not in df.columns:
            raise KeyError(f"Column '{self.text_column}' not found in DataFrame")

        self.df = df.copy()

        texts = df[self.text_column].fillna("").astype(str).tolist()

        self.X_tfidf = self.vectorizer.fit_transform(texts)
        self.feature_names = self.vectorizer.get_feature_names_out()

    def get_matrix(self):
        """
        Return the TF-IDF matrix.
        """
        if self.X_tfidf is None:
            raise RuntimeError("TF-IDF model has not been fitted yet")
        return self.X_tfidf

    def get_terms(self) -> np.ndarray:
        """
        Return the vocabulary terms in column order.
        """
        if self.feature_names is None:
            raise RuntimeError("TF-IDF model has not been fitted yet")
        return self.feature_names

    # --------------------------------------------------
    # Statistics
    # --------------------------------------------------

    def matrix_stats(self) -> Dict[str, float]:
        """
        Return basic statistics of the TF-IDF matrix.
        """
        X = self.get_matrix()

        n_docs, n_terms = X.shape
        density = X.nnz / (n_docs * n_terms)

        return {
            "n_documents": n_docs,
            "n_terms": n_terms,
            "matrix_density": density,
            "matrix_sparsity": 1.0 - density,
        }

    def term_stats(self) -> pd.DataFrame:
        """
        Return term-level statistics (mean TF-IDF).
        """
        X = self.get_matrix()

        mean_tfidf = np.asarray(X.mean(axis=0)).ravel()

        return pd.DataFrame({
            "term": self.get_terms(),
            "mean_tfidf": mean_tfidf
        }).sort_values("mean_tfidf", ascending=False)


