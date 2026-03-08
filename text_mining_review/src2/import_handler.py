# import_handler.py

from pathlib import Path
from collections import Counter
import pandas as pd
import rispy
import bibtexparser


class ImportHandler:
    """
    Handles the import of bibliographic data from multiple sources
    (Scopus, WoS, IEEE, ScienceDirect, ACM, etc.) and unifies
    multiple exports from the same database.
    """

    def __init__(self, file_dict: dict[str, Path]):
        """
        Parameters
        ----------
        file_dict : dict[str, Path]
            Mapping from source name to file path.
            Example:
            {
                "scopus": Path("scopus.csv"),
                "wos": Path("wos.xlsx"),
                "acm_abs": Path("acm1.bib"),
                "acm_kw": Path("acm2.bib")
            }
        """
        self.file_dict = file_dict

    # --------------------------------------------------
    # Public API
    # --------------------------------------------------

    def load_all(self) -> dict[str, pd.DataFrame]:
        """
        Load all files and merge duplicated database exports.

        Returns
        -------
        dict[str, pd.DataFrame]
            Dictionary mapping database name to unified DataFrame.
        """
        raw_dfs = {
            source: self._load_file(path)
            for source, path in self.file_dict.items()
        }

        return self._merge_same_sources(raw_dfs)

    # --------------------------------------------------
    # Internal helpers
    # --------------------------------------------------

    def _load_file(self, path: Path) -> pd.DataFrame:
        """
        Load a single bibliographic file based on its extension.
        """
        suffix = path.suffix.lower()

        if suffix == ".csv":
            return pd.read_csv(path)

        if suffix in {".xls", ".xlsx"}:
            return pd.read_excel(path)

        if suffix == ".bib":
            with open(path, "r", encoding="utf-8") as f:
                entries = bibtexparser.load(f).entries
            return pd.DataFrame(entries)

        if suffix == ".ris":
            with open(path, "r", encoding="utf-8-sig") as f:
                entries = rispy.load(f)
            return pd.DataFrame(entries)

        raise ValueError(f"Unsupported file type: {suffix}")

    def _merge_same_sources(
        self, dfs: dict[str, pd.DataFrame]
    ) -> dict[str, pd.DataFrame]:
        """
        Merge multiple exports from the same database
        (e.g., acm_abs + acm_kw → acm).
        """
        base_names = [key.split("_")[0] for key in dfs.keys()]
        duplicates = {
            name for name, count in Counter(base_names).items() if count > 1
        }

        merged = {}

        for base in set(base_names):
            related_keys = [
                k for k in dfs.keys() if k.startswith(base)
            ]

            if len(related_keys) == 1:
                merged[base] = dfs[related_keys[0]]
            else:
                merged[base] = pd.concat(
                    [dfs[k] for k in related_keys],
                    ignore_index=True
                )

        return merged

