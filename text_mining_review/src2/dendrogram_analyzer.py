# dendrogram_analyzer.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from kneed import KneeLocator
from typing import Optional, Tuple


class DendrogramAnalyzer:
    """
    Performs hierarchical clustering diagnostics:
    - linkage computation
    - dendrogram visualization
    - merge distance analysis
    - semantic elbow detection
    """

    def __init__(
        self,
        X,
        linkage_method: str = "ward",
        metric: str = "euclidean"
    ):
        """
        Parameters
        ----------
        X : array-like or sparse matrix
            Matrix to be clustered (e.g., X_tfidf.T for terms).
        linkage_method : str
            Linkage method (default: 'ward').
        metric : str
            Distance metric (default: 'euclidean').
        """
        self.X = X.toarray() if hasattr(X, "toarray") else X
        self.linkage_method = linkage_method
        self.metric = metric

        self.Z = None
        self.merge_distances = None

    # --------------------------------------------------
    # Core computations
    # --------------------------------------------------

    def compute_linkage(self) -> None:
        """
        Compute the hierarchical linkage matrix.
        """
        self.Z = linkage(
            self.X,
            method=self.linkage_method,
            metric=self.metric
        )
        self.merge_distances = self.Z[:, 2]

    # --------------------------------------------------
    # Visualization
    # --------------------------------------------------

    def plot(
        self,
        truncate_p: int = 30,
        ignore_ratio: float = 0.25,
        figsize: Tuple[int, int] = (14, 6)
    ) -> Tuple[Optional[int], Optional[float]]:
        """
        Plot dendrogram and merge-distance curve with semantic elbow.

        Returns
        -------
        (elbow_step, elbow_distance)
        """
        if self.Z is None:
            self.compute_linkage()

        fig, ax = plt.subplots(1, 2, figsize=figsize)

        # --- Dendrogram ---
        dendrogram(
            self.Z,
            truncate_mode="lastp",
            p=truncate_p,
            leaf_rotation=90,
            leaf_font_size=10,
            ax=ax[0]
        )
        ax[0].set_title("Hierarchical Clustering Dendrogram")
        ax[0].set_ylabel("Ward distance")

        # --- Merge distances ---
        merge_rev = self.merge_distances[::-1]
        steps = np.arange(len(merge_rev))

        ax[1].plot(steps, merge_rev, label="Merge distance")
        ax[1].set_xlabel("Merge step (reversed)")
        ax[1].set_ylabel("Ward distance")
        ax[1].set_title("Merge distances")

        # --- Semantic elbow detection ---
        elbow_step, elbow_dist = self._find_semantic_elbow(
            merge_rev, steps, ignore_ratio
        )

        if elbow_step is not None:
            ax[1].axvline(
                elbow_step,
                color="red",
                linestyle="--",
                label=f"Semantic elbow (step={elbow_step})"
            )
            ax[1].scatter(
                elbow_step,
                elbow_dist,
                color="red",
                zorder=5
            )

        ax[1].legend()
        plt.tight_layout()
        plt.show()

        return elbow_step, elbow_dist

    # --------------------------------------------------
    # Elbow logic
    # --------------------------------------------------

    def _find_semantic_elbow(
        self,
        merge_rev: np.ndarray,
        steps: np.ndarray,
        ignore_ratio: float
    ) -> Tuple[Optional[int], Optional[float]]:
        """
        Detect a semantic elbow while ignoring early noisy merges.
        """
        n = len(merge_rev)
        start = int(n * ignore_ratio)

        if start >= n - 2:
            return None, None

        kl = KneeLocator(
            steps[start:],
            merge_rev[start:],
            curve="convex",
            direction="increasing"
        )

        if kl.knee is None:
            return None, None

        knee_step = int(kl.knee)
        knee_dist = merge_rev[knee_step]

        return knee_step, knee_dist

