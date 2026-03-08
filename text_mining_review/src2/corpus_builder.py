# corpus_builder.py

import pandas as pd
from typing import List


class CorpusBuilder:
    """
    Builds the textual corpus used for vectorization (TF-IDF)
    from a normalized bibliographic DataFrame.
    """

    def __init__(
        self,
        text_fields: List[str] = None,
        output_column: str = "text",
        lowercase: bool = True
    ):
        """
        Parameters
        ----------
        text_fields : list[str]
            Columns to be concatenated to form the corpus text.
            Default: ["title", "abstract", "keywords"]
        output_column : str
            Name of the generated text column.
        lowercase : bool
            Whether to lowercase the text.
        """
        self.text_fields = text_fields or ["title", "abstract", "keywords"]
        self.output_column = output_column
        self.lowercase = lowercase

    # --------------------------------------------------
    # Public API
    # --------------------------------------------------

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build the corpus text column.

        Parameters
        ----------
        df : DataFrame
            Normalized DataFrame.

        Returns
        -------
        DataFrame
            Copy of df with an additional text column.
        """
        df_out = df.copy()

        missing = [c for c in self.text_fields if c not in df_out.columns]
        if missing:
            raise KeyError(f"Missing text fields in DataFrame: {missing}")

        # Replace NaN with empty string
        df_out[self.text_fields] = df_out[self.text_fields].fillna("")

        # Concatenate fields in order
        df_out[self.output_column] = df_out[self.text_fields].astype(str).agg(
            " ".join, axis=1
        )

        if self.lowercase:
            df_out[self.output_column] = df_out[self.output_column].str.lower()

        # Optional: strip extra whitespace
        df_out[self.output_column] = (
            df_out[self.output_column]
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )

        return df_out

