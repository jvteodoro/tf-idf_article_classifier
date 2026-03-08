# normalizer.py

import pandas as pd
from typing import Dict, List, Union


class Normalizer:
    """
    Normalizes bibliographic DataFrames from different databases
    into a unified canonical schema.
    """

    CANONICAL_COLUMNS: List[str] = [
        "title",
        "abstract",
        "keywords",
        "authors",
        "year",
        "doi",
        "source_title",
        "publisher",
        "document_type",
        "database_source",
        "record_id",
    ]

    def __init__(self, column_maps: Dict[str, Dict[str, Union[str, List[str], None]]]):
        """
        Parameters
        ----------
        column_maps : dict
            Mapping from database name to column mapping.
            Example:
            {
                "scopus": {
                    "title": "Title",
                    "abstract": "Abstract",
                    "keywords": ["Author Keywords", "Index Keywords"],
                    ...
                }
            }
        """
        self.column_maps = column_maps

    # --------------------------------------------------
    # Public API
    # --------------------------------------------------

    def normalize_all(
        self, dfs: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        """
        Normalize all database DataFrames.

        Parameters
        ----------
        dfs : dict[str, DataFrame]
            Raw DataFrames keyed by database name.

        Returns
        -------
        dict[str, DataFrame]
            Normalized DataFrames keyed by database name.
        """
        normalized = {}

        for source, df in dfs.items():
            if source not in self.column_maps:
                raise KeyError(f"No column map defined for source '{source}'")

            normalized[source] = self._normalize_single(
                df=df,
                source_name=source,
                column_map=self.column_maps[source],
            )

        return normalized

    def unify(self, dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Concatenate normalized DataFrames into a single DataFrame.

        Empty DataFrames are ignored to avoid dtype issues.
        """
        valid_dfs = [
            df for df in dfs.values()
            if not df.empty and not df.isna().all(axis=None)
        ]

        if not valid_dfs:
            return pd.DataFrame(columns=self.CANONICAL_COLUMNS)

        return pd.concat(valid_dfs, ignore_index=True)

    # --------------------------------------------------
    # Internal helpers
    # --------------------------------------------------

    def _normalize_single(
        self,
        df: pd.DataFrame,
        source_name: str,
        column_map: Dict[str, Union[str, List[str], None]],
    ) -> pd.DataFrame:
        """
        Normalize a single DataFrame into the canonical schema.
        """
        df_norm = df.copy()

        output = {}

        for canonical_col in self.CANONICAL_COLUMNS:
            if canonical_col == "database_source":
                output[canonical_col] = source_name
                continue

            source_col = column_map.get(canonical_col)

            if source_col is None:
                output[canonical_col] = None
                continue

            # Multiple source columns → concatenate
            if isinstance(source_col, list):
                cols = [c for c in source_col if c in df_norm.columns]
                if cols:
                    output[canonical_col] = (
                        df_norm[cols]
                        .astype(str)
                        .agg("; ".join, axis=1)
                    )
                else:
                    output[canonical_col] = None
                continue

            # Single column
            if source_col in df_norm.columns:
                output[canonical_col] = df_norm[source_col]
            else:
                output[canonical_col] = None

        normalized_df = pd.DataFrame(output)

        # Ensure column order
        normalized_df = normalized_df[self.CANONICAL_COLUMNS]

        return normalized_df

