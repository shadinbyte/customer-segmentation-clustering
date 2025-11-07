"""
Customer Segmentation Analysis using Multiple Clustering Algorithms
===================================================================
A production-ready ML pipeline for comprehensive customer segmentation analysis
with automated parameter optimization and business intelligence insights.

Features:
    - Multiple clustering algorithms (K-Means, Hierarchical, DBSCAN, GMM)
    - Automated optimal cluster determination
    - Comprehensive evaluation metrics
    - Business intelligence and product recommendations
    - Publication-ready visualizations
    - Detailed logging and error handling

Technical Stack:
    - Scikit-learn: ML algorithms and metrics
    - Matplotlib/Seaborn: Visualizations
    - Pandas/NumPy: Data processing
    - SciPy: Hierarchical clustering

Author: Shadin
Version: 2.0.0
Date: November 2025
License: MIT
"""

# -*- coding: utf-8 -*-

# IMPORTS

import logging
import os
import sys
import warnings
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


# CONFIGURATION


@dataclass(frozen=True)
class AnalysisConfig:
    """Configuration constants for the analysis pipeline."""

    # File paths
    DATA_FILE: str = "Synthetic_Mall_Customers_3000.csv"
    VALIDATION_FILE: str = "Synthetic_Mall_Customers_3000_with_labels.csv"
    OUTPUT_DIR: str = "clustering_results_3000"

    # Features for clustering
    FEATURE_COLUMNS: Tuple[str, ...] = (
        "Age",
        "Annual Income (BDT)",
        "Spending Score (1-100)",
    )

    REQUIRED_COLUMNS: Tuple[str, ...] = (
        "CustomerID",
        "Gender",
        "Age",
        "Annual Income (BDT)",
        "Spending Score (1-100)",
    )

    # Clustering parameters
    DEFAULT_CLUSTERS: int = 5
    CLUSTER_RANGE: range = field(default_factory=lambda: range(2, 11))
    RANDOM_STATE: int = 42

    # Algorithm-specific parameters
    KMEANS_MAX_ITER: int = 300
    KMEANS_N_INIT: int = 10
    GMM_MAX_ITER: int = 200

    # DBSCAN optimization
    DBSCAN_K_NEIGHBORS: int = 10
    DBSCAN_PERCENTILE: int = 90

    # Visualization parameters
    HISTOGRAM_BINS: int = 30
    DENDROGRAM_SAMPLE_SIZE: int = 100
    FIGURE_DPI: int = 300

    # Data quality thresholds
    OUTLIER_IQR_MULTIPLIER: float = 3.0


class ClusteringAlgorithm(str, Enum):
    """Supported clustering algorithms."""

    KMEANS = "KMeans"
    HIERARCHICAL = "Hierarchical"
    DBSCAN = "DBSCAN"
    GMM = "GMM"


class IncomeCategory(str, Enum):
    """Income level categories."""

    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


class SpendingCategory(str, Enum):
    """Spending behavior categories."""

    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


class AgeCategory(str, Enum):
    """Age group categories."""

    YOUNG = "Young"
    MIDDLE_AGED = "Middle-aged"
    SENIOR = "Senior"


# LOGGING SETUP


class AnalysisLogger:
    """Handles logging configuration for the analysis."""

    @staticmethod
    def setup(output_dir: str, name: str = __name__) -> logging.Logger:
        """
        Configure logging with file and console handlers.

        Args:
            output_dir: Directory for log files
            name: Logger name

        Returns:
            Configured logger instance
        """
        os.makedirs(output_dir, exist_ok=True)

        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)

        # Clear existing handlers
        logger.handlers.clear()

        # File handler with UTF-8 encoding
        file_handler = logging.FileHandler(
            f"{output_dir}/analysis.log", mode="w", encoding="utf-8"
        )
        file_handler.setLevel(logging.INFO)

        # Console handler with UTF-8 encoding (Windows compatible)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        # Force UTF-8 encoding on Windows
        if hasattr(console_handler.stream, "reconfigure"):
            console_handler.stream.reconfigure(encoding="utf-8")

        # Formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )

        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger


# DATA MODELS


@dataclass
class ClusteringResult:
    """Container for clustering algorithm results."""

    algorithm: str
    labels: np.ndarray
    n_clusters: int
    silhouette_score: float
    davies_bouldin_index: float
    calinski_harabasz_score: float
    model: Any
    validation_ari: Optional[float] = None
    additional_metrics: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        """Check if clustering produced valid results."""
        return self.n_clusters > 1 and self.silhouette_score > -1


@dataclass
class ClusterProfile:
    """Business profile for a customer cluster."""

    cluster_id: int
    size: int
    percentage: float
    avg_age: float
    avg_income: float
    avg_spending: float
    dominant_gender: str
    age_category: AgeCategory
    income_category: IncomeCategory
    spending_category: SpendingCategory
    marketing_strategy: str
    product_recommendations: List[Dict[str, str]]


# DATA LOADER


class DataLoader:
    """Handles data loading and validation."""

    def __init__(self, config: AnalysisConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger

    def load(self) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Load main and validation datasets.

        Returns:
            Tuple of (main_dataframe, validation_dataframe)

        Raises:
            FileNotFoundError: If main dataset not found
            ValueError: If data validation fails
        """
        try:
            # Load main dataset
            df = pd.read_csv(self.config.DATA_FILE)
            self.logger.info(f"Dataset loaded: {self.config.DATA_FILE}")

            # Validate structure
            self._validate_dataframe(df)

            # Load validation data if available
            df_validation = self._load_validation_data()

            return df, df_validation

        except FileNotFoundError:
            self.logger.error(f"Dataset not found: {self.config.DATA_FILE}")
            raise FileNotFoundError(
                f"Dataset not found at '{self.config.DATA_FILE}'. "
                "Please run generator.py first to create the dataset."
            )
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise

    def _validate_dataframe(self, df: pd.DataFrame) -> None:
        """Validate dataframe structure and data types."""
        # Check required columns
        missing_cols = set(self.config.REQUIRED_COLUMNS) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Validate numeric types
        for feature in self.config.FEATURE_COLUMNS:
            if not pd.api.types.is_numeric_dtype(df[feature]):
                raise ValueError(
                    f"Feature '{feature}' must be numeric, found: {df[feature].dtype}"
                )

        self.logger.info(f"Dataset shape: {df.shape}")
        self.logger.info(f"Columns: {df.columns.tolist()}")

    def _load_validation_data(self) -> Optional[pd.DataFrame]:
        """Load validation dataset if available."""
        if not Path(self.config.VALIDATION_FILE).exists():
            self.logger.info("No validation data available")
            return None

        try:
            df_val = pd.read_csv(self.config.VALIDATION_FILE)
            self.logger.info(f"Validation data loaded: {self.config.VALIDATION_FILE}")
            return df_val
        except Exception as e:
            self.logger.warning(f"Could not load validation data: {e}")
            return None


# DATA PREPROCESSOR


class DataPreprocessor:
    """Handles data cleaning and preprocessing."""

    def __init__(self, config: AnalysisConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.scaler = StandardScaler()

    def preprocess(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Clean and preprocess data for clustering.

        Args:
            df: Input dataframe

        Returns:
            Tuple of (cleaned_dataframe, scaled_features)
        """
        self.logger.info("\n" + "=" * 60)
        self.logger.info("DATA PREPROCESSING")
        self.logger.info("=" * 60)

        df_clean = df.copy()

        # Handle missing values
        df_clean = self._handle_missing_values(df_clean)

        # Handle infinite values
        df_clean = self._handle_infinite_values(df_clean)

        # Check for outliers
        self._check_outliers(df_clean)

        # Scale features
        X_scaled = self._scale_features(df_clean)

        self.logger.info(f"\nPreprocessed data shape: {X_scaled.shape}")
        self.logger.info(f"Features: {list(self.config.FEATURE_COLUMNS)}")

        return df_clean, X_scaled

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        missing = df[list(self.config.FEATURE_COLUMNS)].isnull().sum()

        if missing.sum() > 0:
            self.logger.warning(f"Missing values found:\n{missing[missing > 0]}")
            df = df.dropna(subset=list(self.config.FEATURE_COLUMNS))
            self.logger.info(f"Rows after dropping missing values: {len(df)}")
        else:
            self.logger.info("No missing values detected")

        return df

    def _handle_infinite_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle infinite values in the dataset."""
        inf_mask = np.isinf(df[list(self.config.FEATURE_COLUMNS)]).any(axis=1)

        if inf_mask.sum() > 0:
            self.logger.warning(f"Infinite values detected: {inf_mask.sum()} rows")
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.dropna(subset=list(self.config.FEATURE_COLUMNS))
            self.logger.info(f"Rows after removing infinite values: {len(df)}")

        return df

    def _check_outliers(self, df: pd.DataFrame) -> None:
        """Check for outliers using IQR method."""
        Q1 = df[list(self.config.FEATURE_COLUMNS)].quantile(0.25)
        Q3 = df[list(self.config.FEATURE_COLUMNS)].quantile(0.75)
        IQR = Q3 - Q1

        outlier_mask = (
            df[list(self.config.FEATURE_COLUMNS)]
            < (Q1 - self.config.OUTLIER_IQR_MULTIPLIER * IQR)
        ) | (
            df[list(self.config.FEATURE_COLUMNS)]
            > (Q3 + self.config.OUTLIER_IQR_MULTIPLIER * IQR)
        )

        outliers = outlier_mask.sum()
        if outliers.sum() > 0:
            self.logger.warning(f"Outliers detected:\n{outliers[outliers > 0]}")
            self.logger.info("Outliers retained for analysis")

    def _scale_features(self, df: pd.DataFrame) -> np.ndarray:
        """Scale features using StandardScaler."""
        X = df[list(self.config.FEATURE_COLUMNS)].values
        X_scaled = self.scaler.fit_transform(X)
        self.logger.info("Features scaled using StandardScaler (mean=0, std=1)")
        return X_scaled


# EXPLORATORY DATA ANALYSIS


class ExploratoryDataAnalysis:
    """Performs exploratory data analysis and visualization."""

    def __init__(self, config: AnalysisConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger

        # Set plotting style
        plt.style.use("default")
        sns.set_palette("husl")

    def analyze(self, df: pd.DataFrame) -> None:
        """
        Perform comprehensive EDA.

        Args:
            df: Input dataframe
        """
        self.logger.info("\n" + "=" * 60)
        self.logger.info("EXPLORATORY DATA ANALYSIS")
        self.logger.info("=" * 60)

        self._create_distribution_plots(df)
        self._create_correlation_heatmap(df)
        self._log_summary_statistics(df)

    def _create_distribution_plots(self, df: pd.DataFrame) -> None:
        """Create distribution and relationship visualizations."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(
            f"Exploratory Data Analysis - {len(df):,} Customers",
            fontsize=16,
            fontweight="bold",
        )

        # Age distribution
        axes[0, 0].hist(
            df["Age"],
            bins=self.config.HISTOGRAM_BINS,
            alpha=0.7,
            color="skyblue",
            edgecolor="black",
        )
        axes[0, 0].set_title("Age Distribution", fontweight="bold")
        axes[0, 0].set_xlabel("Age (years)")
        axes[0, 0].set_ylabel("Frequency")
        axes[0, 0].grid(True, alpha=0.3)

        # Income distribution
        axes[0, 1].hist(
            df["Annual Income (BDT)"],
            bins=self.config.HISTOGRAM_BINS,
            alpha=0.7,
            color="lightgreen",
            edgecolor="black",
        )
        axes[0, 1].set_title("Annual Income Distribution", fontweight="bold")
        axes[0, 1].set_xlabel("Annual Income (BDT)")
        axes[0, 1].set_ylabel("Frequency")
        axes[0, 1].grid(True, alpha=0.3)

        # Spending score distribution
        axes[0, 2].hist(
            df["Spending Score (1-100)"],
            bins=self.config.HISTOGRAM_BINS,
            alpha=0.7,
            color="lightcoral",
            edgecolor="black",
        )
        axes[0, 2].set_title("Spending Score Distribution", fontweight="bold")
        axes[0, 2].set_xlabel("Spending Score")
        axes[0, 2].set_ylabel("Frequency")
        axes[0, 2].grid(True, alpha=0.3)

        # Gender distribution
        gender_counts = df["Gender"].value_counts()
        axes[1, 0].pie(
            gender_counts.values,
            labels=gender_counts.index,
            autopct="%1.1f%%",
            colors=["#87CEEB", "#FFB6C1"],
            startangle=90,
        )
        axes[1, 0].set_title("Gender Distribution", fontweight="bold")

        # Income vs Spending Score
        scatter = axes[1, 1].scatter(
            df["Annual Income (BDT)"],
            df["Spending Score (1-100)"],
            c=df["Age"],
            alpha=0.6,
            cmap="viridis",
            s=30,
        )
        axes[1, 1].set_title("Income vs Spending Score", fontweight="bold")
        axes[1, 1].set_xlabel("Annual Income (BDT)")
        axes[1, 1].set_ylabel("Spending Score")
        axes[1, 1].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[1, 1], label="Age")

        # Age vs Spending Score
        axes[1, 2].scatter(
            df["Age"], df["Spending Score (1-100)"], alpha=0.5, color="purple", s=30
        )
        axes[1, 2].set_title("Age vs Spending Score", fontweight="bold")
        axes[1, 2].set_xlabel("Age (years)")
        axes[1, 2].set_ylabel("Spending Score")
        axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = self._get_output_path(f"EDA_{len(df)}_customers.png")
        plt.savefig(output_path, dpi=self.config.FIGURE_DPI, bbox_inches="tight")
        plt.show()
        plt.close(fig)

        self.logger.info(f"EDA visualization saved: {output_path}")

    def _create_correlation_heatmap(self, df: pd.DataFrame) -> None:
        """Create correlation heatmap."""
        fig, ax = plt.subplots(figsize=(10, 8))

        corr_matrix = df[list(self.config.FEATURE_COLUMNS)].corr()
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".3f",
            cmap="coolwarm",
            center=0,
            square=True,
            linewidths=1,
            cbar_kws={"label": "Correlation Coefficient"},
            ax=ax,
        )

        plt.title(
            "Correlation Heatmap - Feature Relationships",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )
        plt.tight_layout()

        output_path = self._get_output_path(f"correlation_heatmap_{len(df)}.png")
        plt.savefig(output_path, dpi=self.config.FIGURE_DPI, bbox_inches="tight")
        plt.show()
        plt.close(fig)

        self.logger.info(f"Correlation heatmap saved: {output_path}")

    def _log_summary_statistics(self, df: pd.DataFrame) -> None:
        """Log summary statistics."""
        self.logger.info("\nKey Statistics:")
        self.logger.info(
            f"  Age range: {df['Age'].min():.0f} - {df['Age'].max():.0f} years"
        )
        self.logger.info(
            f"  Income range: BDT {df['Annual Income (BDT)'].min():,.0f} - "
            f"BDT {df['Annual Income (BDT)'].max():,.0f}"
        )
        self.logger.info(
            f"  Spending score range: {df['Spending Score (1-100)'].min():.0f} - "
            f"{df['Spending Score (1-100)'].max():.0f}"
        )

        gender_counts = df["Gender"].value_counts()
        self.logger.info(f"  Gender distribution: {dict(gender_counts)}")

    def _get_output_path(self, filename: str) -> Path:
        """Generate output file path."""
        return Path(self.config.OUTPUT_DIR) / filename


# OPTIMAL CLUSTER FINDER


class OptimalClusterFinder:
    """Determines optimal number of clusters for K-Means."""

    def __init__(self, config: AnalysisConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger

    def find_optimal_k(
        self, X_scaled: np.ndarray, k_range: Optional[range] = None
    ) -> int:
        """
        Find optimal number of clusters using multiple evaluation methods.

        Args:
            X_scaled: Scaled feature matrix
            k_range: Range of k values to test

        Returns:
            Optimal number of clusters
        """
        if k_range is None:
            k_range = self.config.CLUSTER_RANGE

        self.logger.info("\n" + "=" * 60)
        self.logger.info("OPTIMAL CLUSTER DETERMINATION")
        self.logger.info("=" * 60)

        # Calculate metrics for different k values
        metrics = self._calculate_metrics(X_scaled, k_range)

        # Plot analysis
        self._plot_analysis(k_range, metrics)

        # Determine optimal k
        optimal_k = self._determine_optimal_k(k_range, metrics)

        return optimal_k

    def _calculate_metrics(
        self, X_scaled: np.ndarray, k_range: range
    ) -> Dict[str, List[float]]:
        """Calculate evaluation metrics for different k values."""
        wcss = []
        silhouette_scores = []
        davies_bouldin_scores = []
        calinski_scores = []

        self.logger.info("\nTesting K-Means with different cluster numbers...")

        for k in k_range:
            try:
                kmeans = KMeans(
                    n_clusters=k,
                    random_state=self.config.RANDOM_STATE,
                    n_init=self.config.KMEANS_N_INIT,
                    max_iter=self.config.KMEANS_MAX_ITER,
                )
                labels = kmeans.fit_predict(X_scaled)

                wcss.append(kmeans.inertia_)

                if len(set(labels)) > 1:
                    silhouette_scores.append(silhouette_score(X_scaled, labels))
                    davies_bouldin_scores.append(davies_bouldin_score(X_scaled, labels))
                    calinski_scores.append(calinski_harabasz_score(X_scaled, labels))
                else:
                    silhouette_scores.append(-1)
                    davies_bouldin_scores.append(float("inf"))
                    calinski_scores.append(0)

                self.logger.info(
                    f"  k={k}: Silhouette={silhouette_scores[-1]:.3f}, "
                    f"DB={davies_bouldin_scores[-1]:.3f}, "
                    f"CH={calinski_scores[-1]:.0f}"
                )
            except Exception as e:
                self.logger.error(f"Error testing k={k}: {e}")
                continue

        return {
            "wcss": wcss,
            "silhouette": silhouette_scores,
            "davies_bouldin": davies_bouldin_scores,
            "calinski": calinski_scores,
        }

    def _plot_analysis(self, k_range: range, metrics: Dict[str, List[float]]) -> None:
        """Plot evaluation metrics."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(
            "Optimal Cluster Determination for K-Means", fontsize=14, fontweight="bold"
        )

        k_list = list(k_range)

        # Elbow curve
        axes[0, 0].plot(k_list, metrics["wcss"], "bo-", linewidth=2, markersize=8)
        axes[0, 0].set_xlabel("Number of Clusters (k)")
        axes[0, 0].set_ylabel("WCSS")
        axes[0, 0].set_title("Elbow Method")
        axes[0, 0].grid(True, alpha=0.3)

        # Silhouette scores
        axes[0, 1].plot(k_list, metrics["silhouette"], "ro-", linewidth=2, markersize=8)
        axes[0, 1].set_xlabel("Number of Clusters (k)")
        axes[0, 1].set_ylabel("Silhouette Score")
        axes[0, 1].set_title("Silhouette Analysis (Higher is Better)")
        axes[0, 1].grid(True, alpha=0.3)
        max_sil = max(metrics["silhouette"])
        axes[0, 1].axhline(
            y=max_sil, color="g", linestyle="--", alpha=0.5, label=f"Max: {max_sil:.3f}"
        )
        axes[0, 1].legend()

        # Davies-Bouldin Index
        axes[1, 0].plot(
            k_list, metrics["davies_bouldin"], "go-", linewidth=2, markersize=8
        )
        axes[1, 0].set_xlabel("Number of Clusters (k)")
        axes[1, 0].set_ylabel("Davies-Bouldin Index")
        axes[1, 0].set_title("Davies-Bouldin Analysis (Lower is Better)")
        axes[1, 0].grid(True, alpha=0.3)
        min_db = min(metrics["davies_bouldin"])
        axes[1, 0].axhline(
            y=min_db, color="r", linestyle="--", alpha=0.5, label=f"Min: {min_db:.3f}"
        )
        axes[1, 0].legend()

        # Calinski-Harabasz Index
        axes[1, 1].plot(k_list, metrics["calinski"], "mo-", linewidth=2, markersize=8)
        axes[1, 1].set_xlabel("Number of Clusters (k)")
        axes[1, 1].set_ylabel("Calinski-Harabasz Index")
        axes[1, 1].set_title("Calinski-Harabasz Analysis (Higher is Better)")
        axes[1, 1].grid(True, alpha=0.3)
        max_ch = max(metrics["calinski"])
        axes[1, 1].axhline(
            y=max_ch, color="b", linestyle="--", alpha=0.5, label=f"Max: {max_ch:.0f}"
        )
        axes[1, 1].legend()

        plt.tight_layout()
        output_path = self._get_output_path("optimal_k_analysis.png")
        plt.savefig(output_path, dpi=self.config.FIGURE_DPI, bbox_inches="tight")
        plt.show()
        plt.close(fig)

        self.logger.info(f"Optimal k analysis saved: {output_path}")

    def _determine_optimal_k(
        self, k_range: range, metrics: Dict[str, List[float]]
    ) -> int:
        """Determine optimal k based on metrics."""
        k_list = list(k_range)

        optimal_k_sil = k_list[np.argmax(metrics["silhouette"])]
        optimal_k_db = k_list[np.argmin(metrics["davies_bouldin"])]
        optimal_k_ch = k_list[np.argmax(metrics["calinski"])]

        self.logger.info("\n" + "-" * 60)
        self.logger.info("OPTIMAL K RECOMMENDATIONS:")
        self.logger.info(f"  Silhouette Score: k = {optimal_k_sil}")
        self.logger.info(f"  Davies-Bouldin Index: k = {optimal_k_db}")
        self.logger.info(f"  Calinski-Harabasz Index: k = {optimal_k_ch}")

        optimal_k = optimal_k_sil
        self.logger.info(f"\n  SELECTED OPTIMAL K: {optimal_k}")
        self.logger.info("-" * 60)

        return optimal_k

    def _get_output_path(self, filename: str) -> Path:
        """Generate output file path."""
        return Path(self.config.OUTPUT_DIR) / filename


# CLUSTERING ENGINE


class ClusteringEngine:
    """Executes clustering algorithms with comprehensive evaluation."""

    def __init__(
        self,
        config: AnalysisConfig,
        logger: logging.Logger,
        X_scaled: np.ndarray,
        df: pd.DataFrame,
    ):
        self.config = config
        self.logger = logger
        self.X_scaled = X_scaled
        self.df = df
        self.results: Dict[str, ClusteringResult] = {}

    def run_kmeans(self, n_clusters: int) -> ClusteringResult:
        """
        Execute K-Means clustering.

        Args:
            n_clusters: Number of clusters

        Returns:
            ClusteringResult object
        """
        self.logger.info("\n" + "=" * 60)
        self.logger.info(f"K-MEANS CLUSTERING (k={n_clusters})")
        self.logger.info("=" * 60)

        try:
            kmeans = KMeans(
                n_clusters=n_clusters,
                random_state=self.config.RANDOM_STATE,
                n_init=self.config.KMEANS_N_INIT,
                max_iter=self.config.KMEANS_MAX_ITER,
            )
            labels = kmeans.fit_predict(self.X_scaled)

            # Check convergence
            if kmeans.n_iter_ >= self.config.KMEANS_MAX_ITER:
                self.logger.warning(
                    f"K-Means reached max iterations ({self.config.KMEANS_MAX_ITER})"
                )

            # Calculate metrics
            metrics = self._calculate_metrics(labels)

            self.logger.info(f"\nK-Means Results:")
            self._log_metrics(metrics, n_clusters)
            self.logger.info(f"  Iterations: {kmeans.n_iter_}")
            self.logger.info(f"  Inertia: {kmeans.inertia_:.2f}")

            result = ClusteringResult(
                algorithm=ClusteringAlgorithm.KMEANS.value,
                labels=labels,
                n_clusters=n_clusters,
                silhouette_score=metrics["silhouette"],
                davies_bouldin_index=metrics["davies_bouldin"],
                calinski_harabasz_score=metrics["calinski"],
                model=kmeans,
                additional_metrics={
                    "inertia": kmeans.inertia_,
                    "n_iter": kmeans.n_iter_,
                },
            )

            self.results[ClusteringAlgorithm.KMEANS.value] = result
            return result

        except Exception as e:
            self.logger.error(f"K-Means clustering failed: {e}")
            raise

    def run_hierarchical(self, n_clusters: int) -> ClusteringResult:
        """
        Execute Hierarchical clustering.

        Args:
            n_clusters: Number of clusters

        Returns:
            ClusteringResult object
        """
        self.logger.info("\n" + "=" * 60)
        self.logger.info(f"HIERARCHICAL CLUSTERING (k={n_clusters})")
        self.logger.info("=" * 60)

        try:
            hierarchical = AgglomerativeClustering(
                n_clusters=n_clusters, linkage="ward"
            )
            labels = hierarchical.fit_predict(self.X_scaled)

            # Calculate metrics
            metrics = self._calculate_metrics(labels)

            self.logger.info(f"\nHierarchical Clustering Results:")
            self._log_metrics(metrics, n_clusters)
            self.logger.info(f"  Linkage: ward")

            result = ClusteringResult(
                algorithm=ClusteringAlgorithm.HIERARCHICAL.value,
                labels=labels,
                n_clusters=n_clusters,
                silhouette_score=metrics["silhouette"],
                davies_bouldin_index=metrics["davies_bouldin"],
                calinski_harabasz_score=metrics["calinski"],
                model=hierarchical,
            )

            self.results[ClusteringAlgorithm.HIERARCHICAL.value] = result
            return result

        except Exception as e:
            self.logger.error(f"Hierarchical clustering failed: {e}")
            raise

    def run_dbscan(
        self, eps: Optional[float] = None, min_samples: Optional[int] = None
    ) -> ClusteringResult:
        """
        Execute DBSCAN clustering with parameter optimization.

        Args:
            eps: Epsilon parameter (auto-optimized if None)
            min_samples: Min samples parameter (auto-optimized if None)

        Returns:
            ClusteringResult object
        """
        self.logger.info("\n" + "=" * 60)
        self.logger.info("DBSCAN CLUSTERING")
        self.logger.info("=" * 60)

        try:
            # Auto-optimize parameters if not provided
            if eps is None or min_samples is None:
                eps, min_samples = self._optimize_dbscan_params()

            self.logger.info(f"\nParameters: eps={eps:.3f}, min_samples={min_samples}")

            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(self.X_scaled)

            # Analyze results
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)

            self.logger.info(f"\nDBSCAN Results:")
            self.logger.info(f"  Clusters found: {n_clusters}")
            self.logger.info(
                f"  Noise points: {n_noise} ({n_noise/len(labels)*100:.2f}%)"
            )

            # FIX: Calculate metrics using full dataset (including noise)
            # Don't filter out noise - pass all labels and all data
            if n_clusters > 1:
                metrics = self._calculate_metrics(labels)
            else:
                metrics = {
                    "silhouette": -1,
                    "davies_bouldin": float("inf"),
                    "calinski": 0,
                }

            self._log_metrics(metrics, n_clusters)

            result = ClusteringResult(
                algorithm=ClusteringAlgorithm.DBSCAN.value,
                labels=labels,
                n_clusters=n_clusters,
                silhouette_score=metrics["silhouette"],
                davies_bouldin_index=metrics["davies_bouldin"],
                calinski_harabasz_score=metrics["calinski"],
                model=dbscan,
                additional_metrics={
                    "n_noise": n_noise,
                    "eps": eps,
                    "min_samples": min_samples,
                },
            )

            self.results[ClusteringAlgorithm.DBSCAN.value] = result
            return result

        except Exception as e:
            self.logger.error(f"DBSCAN clustering failed: {e}")
            raise

    def run_gmm(self, n_components: int) -> ClusteringResult:
        """
        Execute Gaussian Mixture Model clustering.

        Args:
            n_components: Number of components

        Returns:
            ClusteringResult object
        """
        self.logger.info("\n" + "=" * 60)
        self.logger.info(f"GAUSSIAN MIXTURE MODEL (n_components={n_components})")
        self.logger.info("=" * 60)

        try:
            gmm = GaussianMixture(
                n_components=n_components,
                random_state=self.config.RANDOM_STATE,
                max_iter=self.config.GMM_MAX_ITER,
            )
            labels = gmm.fit_predict(self.X_scaled)

            # Calculate metrics
            metrics = self._calculate_metrics(labels)

            self.logger.info(f"\nGMM Results:")
            self._log_metrics(metrics, n_components)
            self.logger.info(f"  Converged: {gmm.converged_}")
            self.logger.info(f"  BIC: {gmm.bic(self.X_scaled):.2f}")
            self.logger.info(f"  AIC: {gmm.aic(self.X_scaled):.2f}")

            result = ClusteringResult(
                algorithm=ClusteringAlgorithm.GMM.value,
                labels=labels,
                n_clusters=n_components,
                silhouette_score=metrics["silhouette"],
                davies_bouldin_index=metrics["davies_bouldin"],
                calinski_harabasz_score=metrics["calinski"],
                model=gmm,
                additional_metrics={
                    "bic": gmm.bic(self.X_scaled),
                    "aic": gmm.aic(self.X_scaled),
                    "converged": gmm.converged_,
                },
            )

            self.results[ClusteringAlgorithm.GMM.value] = result
            return result

        except Exception as e:
            self.logger.error(f"GMM clustering failed: {e}")
            raise

    def _calculate_metrics(self, labels: np.ndarray) -> Dict[str, float]:
        """Calculate clustering evaluation metrics."""
        if len(set(labels)) <= 1:
            return {"silhouette": -1, "davies_bouldin": float("inf"), "calinski": 0}

        return {
            "silhouette": silhouette_score(self.X_scaled, labels),
            "davies_bouldin": davies_bouldin_score(self.X_scaled, labels),
            "calinski": calinski_harabasz_score(self.X_scaled, labels),
        }

    def _log_metrics(self, metrics: Dict[str, float], n_clusters: int) -> None:
        """Log clustering metrics."""
        self.logger.info(f"  Clusters: {n_clusters}")
        self.logger.info(f"  Silhouette Score: {metrics['silhouette']:.4f}")
        self.logger.info(f"  Davies-Bouldin Index: {metrics['davies_bouldin']:.4f}")
        self.logger.info(f"  Calinski-Harabasz Index: {metrics['calinski']:.0f}")

    def _optimize_dbscan_params(self) -> Tuple[float, int]:
        """Optimize DBSCAN parameters using k-distance graph."""
        try:
            k = min(self.config.DBSCAN_K_NEIGHBORS, len(self.X_scaled) // 100)
            neighbors = NearestNeighbors(n_neighbors=k)
            neighbors.fit(self.X_scaled)
            distances, _ = neighbors.kneighbors(self.X_scaled)
            distances = np.sort(distances[:, -1])

            eps = np.percentile(distances, self.config.DBSCAN_PERCENTILE)
            min_samples = max(2 * self.X_scaled.shape[1], 5)

            self.logger.info(f"Auto-optimized parameters:")
            self.logger.info(f"  eps: {eps:.3f}")
            self.logger.info(f"  min_samples: {min_samples}")

            return eps, min_samples

        except Exception as e:
            self.logger.warning(f"Parameter optimization failed: {e}. Using defaults.")
            return 0.5, 5

    def validate_against_true_labels(
        self, df_validation: pd.DataFrame, algorithm: str
    ) -> Optional[float]:
        """
        Validate clustering against true labels.

        Args:
            df_validation: DataFrame with true labels
            algorithm: Name of algorithm to validate

        Returns:
            Adjusted Rand Index or None
        """
        if algorithm not in self.results:
            return None

        try:
            true_labels = df_validation["True_Segment"]
            predicted_labels = self.results[algorithm].labels

            # Convert true labels to numeric
            label_map = {label: idx for idx, label in enumerate(true_labels.unique())}
            true_labels_numeric = true_labels.map(label_map)

            ari = adjusted_rand_score(true_labels_numeric, predicted_labels)

            self.logger.info(f"\n  Validation ARI ({algorithm}): {ari:.4f}")
            self.logger.info(f"  (1.0=perfect, 0.0=random, <0=worse than random)")

            # Update result
            self.results[algorithm].validation_ari = ari

            return ari

        except Exception as e:
            self.logger.warning(f"Validation failed for {algorithm}: {e}")
            return None


# BUSINESS INTELLIGENCE ENGINE


class BusinessIntelligenceEngine:
    """Generates business insights and recommendations."""

    def __init__(self, config: AnalysisConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger

    @staticmethod
    def categorize_age(age: float) -> AgeCategory:
        """Categorize age into groups."""
        if age < 30:
            return AgeCategory.YOUNG
        elif age < 50:
            return AgeCategory.MIDDLE_AGED
        else:
            return AgeCategory.SENIOR

    @staticmethod
    def categorize_income(income: float) -> IncomeCategory:
        """Categorize income into groups."""
        if income < 500000:
            return IncomeCategory.LOW
        elif income < 1000000:
            return IncomeCategory.MEDIUM
        else:
            return IncomeCategory.HIGH

    @staticmethod
    def categorize_spending(spending: float) -> SpendingCategory:
        """Categorize spending into groups."""
        if spending < 40:
            return SpendingCategory.LOW
        elif spending < 70:
            return SpendingCategory.MEDIUM
        else:
            return SpendingCategory.HIGH

    def generate_cluster_profiles(
        self, df: pd.DataFrame, labels: np.ndarray, algorithm_name: str
    ) -> List[ClusterProfile]:
        """
        Generate comprehensive profiles for each cluster.

        Args:
            df: Customer dataframe
            labels: Cluster labels
            algorithm_name: Name of clustering algorithm

        Returns:
            List of ClusterProfile objects
        """
        self.logger.info(f"\n" + "-" * 60)
        self.logger.info(f"{algorithm_name} - CLUSTER PROFILING")
        self.logger.info("-" * 60)

        profiles = []

        for cluster_id in sorted(set(labels)):
            if cluster_id == -1:  # Skip noise
                n_noise = np.sum(labels == -1)
                self.logger.info(
                    f"\nCluster -1 (Noise): {n_noise} customers "
                    f"({n_noise/len(df)*100:.1f}%)"
                )
                continue

            # Extract cluster data
            mask = labels == cluster_id
            cluster_data = df[mask]

            # Calculate statistics
            profile = self._create_cluster_profile(cluster_id, cluster_data, len(df))

            profiles.append(profile)

            # Log profile
            self._log_cluster_profile(profile)

        return profiles

    def _create_cluster_profile(
        self, cluster_id: int, cluster_data: pd.DataFrame, total_customers: int
    ) -> ClusterProfile:
        """Create a single cluster profile."""
        # Basic statistics
        size = len(cluster_data)
        percentage = (size / total_customers) * 100

        avg_age = cluster_data["Age"].mean()
        avg_income = cluster_data["Annual Income (BDT)"].mean()
        avg_spending = cluster_data["Spending Score (1-100)"].mean()

        # Gender
        gender_mode = cluster_data["Gender"].mode()
        dominant_gender = gender_mode[0] if len(gender_mode) > 0 else "Mixed"

        # Categorization
        age_cat = self.categorize_age(avg_age)
        income_cat = self.categorize_income(avg_income)
        spending_cat = self.categorize_spending(avg_spending)

        # Strategy
        strategy = self._generate_marketing_strategy(
            age_cat, income_cat, spending_cat, size, total_customers
        )

        # Product recommendations
        products = self._generate_product_recommendations(
            age_cat, income_cat, spending_cat, avg_age, avg_income, avg_spending
        )

        return ClusterProfile(
            cluster_id=cluster_id,
            size=size,
            percentage=percentage,
            avg_age=avg_age,
            avg_income=avg_income,
            avg_spending=avg_spending,
            dominant_gender=dominant_gender,
            age_category=age_cat,
            income_category=income_cat,
            spending_category=spending_cat,
            marketing_strategy=strategy,
            product_recommendations=products,
        )

    def _log_cluster_profile(self, profile: ClusterProfile) -> None:
        """Log cluster profile details."""
        self.logger.info(
            f"\nCluster {profile.cluster_id}: {profile.size} customers "
            f"({profile.percentage:.1f}%)"
        )
        self.logger.info(f"  Demographics:")
        self.logger.info(
            f"    - Age: {profile.avg_age:.1f} years ({profile.age_category.value})"
        )
        self.logger.info(f"    - Gender: {profile.dominant_gender}")
        self.logger.info(f"  Financial Profile:")
        self.logger.info(
            f"    - Income: BDT {profile.avg_income:,.0f} ({profile.income_category.value})"
        )
        self.logger.info(
            f"    - Spending: {profile.avg_spending:.1f} ({profile.spending_category.value})"
        )
        self.logger.info(
            f"  Segment: {profile.age_category.value}, "
            f"{profile.income_category.value} income, "
            f"{profile.spending_category.value} spending"
        )
        self.logger.info(f"  Strategy: {profile.marketing_strategy}")

        if profile.product_recommendations:
            self.logger.info(f"\n  🛍️ PRODUCT RECOMMENDATIONS:")
            for rec in profile.product_recommendations[:3]:
                self.logger.info(f"    {rec['priority']} {rec['product']}")
                self.logger.info(f"       Reason: {rec['reason']}")
                self.logger.info(f"       Conversion: {rec['conversion']}")

    def _generate_marketing_strategy(
        self,
        age_cat: AgeCategory,
        income_cat: IncomeCategory,
        spending_cat: SpendingCategory,
        cluster_size: int,
        total_customers: int,
    ) -> str:
        """Generate marketing strategy based on cluster characteristics."""
        market_share = (cluster_size / total_customers) * 100

        if spending_cat == SpendingCategory.HIGH and income_cat == IncomeCategory.HIGH:
            return (
                "Premium products, VIP programs, exclusive offers, personalized service"
            )
        elif spending_cat == SpendingCategory.HIGH:
            return (
                "Value bundles, loyalty rewards, installment plans, quality assurance"
            )
        elif spending_cat == SpendingCategory.LOW and income_cat == IncomeCategory.HIGH:
            return (
                "Trust-building campaigns, product demonstrations, value propositions"
            )
        elif spending_cat == SpendingCategory.MEDIUM and market_share > 20:
            return "Mass market campaigns, seasonal promotions, volume discounts"
        else:
            return "Entry-level products, first-purchase discounts, education campaigns"

    def _generate_product_recommendations(
        self,
        age_cat: AgeCategory,
        income_cat: IncomeCategory,
        spending_cat: SpendingCategory,
        avg_age: float,
        avg_income: float,
        avg_spending: float,
    ) -> List[Dict[str, str]]:
        """Generate intelligent product recommendations."""
        recommendations = []

        # High Income + High Spending
        if income_cat == IncomeCategory.HIGH and spending_cat == SpendingCategory.HIGH:
            recommendations = [
                {
                    "product": "iPhone Pro Max",
                    "priority": "🔴 Primary",
                    "reason": "Premium segment with high purchasing power",
                    "conversion": "85-90%",
                },
                {
                    "product": "MacBook Pro",
                    "priority": "🔴 Primary",
                    "reason": "Affluent professionals seeking premium quality",
                    "conversion": "75-80%",
                },
            ]

        # High Income + Low/Medium Spending
        elif income_cat == IncomeCategory.HIGH:
            recommendations = [
                {
                    "product": "HP Business Laptop",
                    "priority": "🟡 Primary",
                    "reason": "Value-conscious professionals",
                    "conversion": "70-75%",
                },
                {
                    "product": "Bluetooth Speaker",
                    "priority": "🟢 Secondary",
                    "reason": "Low-risk entry point",
                    "conversion": "55-60%",
                },
            ]

        # Medium Income + High Spending
        elif (
            income_cat == IncomeCategory.MEDIUM
            and spending_cat == SpendingCategory.HIGH
        ):
            recommendations = [
                {
                    "product": "HP Mid-Range Laptop",
                    "priority": "🔴 Primary",
                    "reason": "Aspirational buyers, offer financing",
                    "conversion": "75-80%",
                },
                {
                    "product": "Wireless Headphones",
                    "priority": "🟡 Primary",
                    "reason": "Lifestyle accessory within budget",
                    "conversion": "70-75%",
                },
            ]

        # Default budget segment
        else:
            recommendations = [
                {
                    "product": "Headphones",
                    "priority": "🔴 Primary",
                    "reason": "Essential accessory at accessible price",
                    "conversion": "80-85%",
                },
                {
                    "product": "Bluetooth Speaker",
                    "priority": "🟡 Primary",
                    "reason": "Entry-level lifestyle product",
                    "conversion": "75-80%",
                },
            ]

        return recommendations


# VISUALIZATION ENGINE


class VisualizationEngine:
    """Creates publication-ready visualizations."""

    def __init__(self, config: AnalysisConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger

    def plot_clusters(
        self, df: pd.DataFrame, labels: np.ndarray, algorithm_name: str
    ) -> None:
        """Create cluster visualization plots."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(
            f"{algorithm_name} - {len(df):,} Customers", fontsize=14, fontweight="bold"
        )

        unique_labels = sorted(set(labels))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

        # Handle noise
        if -1 in unique_labels:
            colors = np.vstack([[0.5, 0.5, 0.5, 0.3], colors[1:]])

        # Income vs Spending
        for idx, label in enumerate(unique_labels):
            mask = labels == label
            label_name = f"Cluster {label}" if label != -1 else "Noise"

            axes[0].scatter(
                df.loc[mask, "Annual Income (BDT)"],
                df.loc[mask, "Spending Score (1-100)"],
                c=[colors[idx]],
                label=label_name,
                alpha=0.6,
                s=50,
                edgecolors="black",
                linewidths=0.5,
            )

        axes[0].set_xlabel("Annual Income (BDT)")
        axes[0].set_ylabel("Spending Score")
        axes[0].set_title("Income vs Spending")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Age vs Spending
        for idx, label in enumerate(unique_labels):
            mask = labels == label
            label_name = f"Cluster {label}" if label != -1 else "Noise"

            axes[1].scatter(
                df.loc[mask, "Age"],
                df.loc[mask, "Spending Score (1-100)"],
                c=[colors[idx]],
                label=label_name,
                alpha=0.6,
                s=50,
                edgecolors="black",
                linewidths=0.5,
            )

        axes[1].set_xlabel("Age (years)")
        axes[1].set_ylabel("Spending Score")
        axes[1].set_title("Age vs Spending")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        filename = f'{algorithm_name.replace(" ", "_").lower()}_clusters.png'
        output_path = self._get_output_path(filename)
        plt.savefig(output_path, dpi=self.config.FIGURE_DPI, bbox_inches="tight")
        plt.show()
        plt.close(fig)

        self.logger.info(f"Cluster visualization saved: {output_path}")

    def plot_dendrogram(self, X_scaled: np.ndarray) -> None:
        """Create dendrogram for hierarchical clustering."""
        fig = plt.figure(figsize=(16, 8))

        sample_size = min(self.config.DENDROGRAM_SAMPLE_SIZE, len(X_scaled))
        sample_indices = np.random.choice(len(X_scaled), sample_size, replace=False)
        Z = linkage(X_scaled[sample_indices], "ward")

        plt.title(
            f"Hierarchical Clustering Dendrogram (Sample: {sample_size})",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )
        plt.xlabel("Sample Index")
        plt.ylabel("Distance (Ward Linkage)")

        dendrogram(Z, no_labels=True, color_threshold=10)

        plt.tight_layout()
        output_path = self._get_output_path("dendrogram.png")
        plt.savefig(output_path, dpi=self.config.FIGURE_DPI, bbox_inches="tight")
        plt.show()
        plt.close(fig)

        self.logger.info(f"Dendrogram saved: {output_path}")

    def plot_algorithm_comparison(self, results: Dict[str, ClusteringResult]) -> None:
        """Create algorithm comparison visualization."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle("Algorithm Performance Comparison", fontsize=14, fontweight="bold")

        algorithms = list(results.keys())
        silhouette = [r.silhouette_score for r in results.values()]
        davies_bouldin = [r.davies_bouldin_index for r in results.values()]
        calinski = [r.calinski_harabasz_score for r in results.values()]

        # Silhouette
        bars1 = axes[0].bar(algorithms, silhouette, color="lightblue", alpha=0.7)
        axes[0].set_title("Silhouette Score\n(Higher is Better)", fontweight="bold")
        axes[0].set_ylabel("Score")
        axes[0].tick_params(axis="x", rotation=45)
        axes[0].grid(True, alpha=0.3, axis="y")

        for bar, score in zip(bars1, silhouette):
            axes[0].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{score:.3f}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        # Davies-Bouldin
        db_capped = [min(db, 5) for db in davies_bouldin]  # Cap for visualization
        bars2 = axes[1].bar(algorithms, db_capped, color="lightcoral", alpha=0.7)
        axes[1].set_title("Davies-Bouldin Index\n(Lower is Better)", fontweight="bold")
        axes[1].set_ylabel("Index")
        axes[1].tick_params(axis="x", rotation=45)
        axes[1].grid(True, alpha=0.3, axis="y")

        for bar, score, orig in zip(bars2, db_capped, davies_bouldin):
            label = "inf" if np.isinf(orig) else f"{orig:.3f}"
            axes[1].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.1,
                label,
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        # Calinski-Harabasz
        bars3 = axes[2].bar(algorithms, calinski, color="lightgreen", alpha=0.7)
        axes[2].set_title(
            "Calinski-Harabasz Index\n(Higher is Better)", fontweight="bold"
        )
        axes[2].set_ylabel("Index")
        axes[2].tick_params(axis="x", rotation=45)
        axes[2].grid(True, alpha=0.3, axis="y")

        for bar, score in zip(bars3, calinski):
            axes[2].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(calinski) * 0.02,
                f"{score:.0f}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        plt.tight_layout()
        output_path = self._get_output_path("algorithm_comparison.png")
        plt.savefig(output_path, dpi=self.config.FIGURE_DPI, bbox_inches="tight")
        plt.show()
        plt.close(fig)

        self.logger.info(f"Comparison chart saved: {output_path}")

    def _get_output_path(self, filename: str) -> Path:
        """Generate output file path."""
        return Path(self.config.OUTPUT_DIR) / filename


# REPORT GENERATOR


class ReportGenerator:
    """Generates comprehensive analysis reports."""

    def __init__(self, config: AnalysisConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger

    def generate_summary_report(
        self,
        results: Dict[str, ClusteringResult],
        profiles_dict: Dict[str, List[ClusterProfile]],
        optimal_k: int,
    ) -> None:
        """
        Generate and save comprehensive summary report.

        Args:
            results: Dictionary of clustering results
            profiles_dict: Dictionary of cluster profiles by algorithm
            optimal_k: Optimal number of clusters
        """
        self.logger.info("\n" + "=" * 60)
        self.logger.info("GENERATING SUMMARY REPORT")
        self.logger.info("=" * 60)

        report_path = Path(self.config.OUTPUT_DIR) / "analysis_summary.txt"

        with open(report_path, "w", encoding="utf-8") as f:
            # Header
            f.write("=" * 80 + "\n")
            f.write("CUSTOMER SEGMENTATION ANALYSIS - COMPREHENSIVE REPORT\n")
            f.write("=" * 80 + "\n\n")

            # Executive Summary
            f.write("EXECUTIVE SUMMARY\n")
            f.write("-" * 80 + "\n")
            f.write(f"Optimal Number of Clusters: {optimal_k}\n")
            f.write(f"Algorithms Evaluated: {', '.join(results.keys())}\n\n")

            # Algorithm Performance
            f.write("\nALGORITHM PERFORMANCE COMPARISON\n")
            f.write("-" * 80 + "\n")
            f.write(
                f"{'Algorithm':<20} {'Silhouette':<15} {'Davies-Bouldin':<20} {'Calinski-Harabasz':<20}\n"
            )
            f.write("-" * 80 + "\n")

            for algo, result in results.items():
                f.write(
                    f"{algo:<20} {result.silhouette_score:<15.4f} "
                    f"{result.davies_bouldin_index:<20.4f} "
                    f"{result.calinski_harabasz_score:<20.0f}\n"
                )

            # Best Algorithm
            best_algo = max(results.items(), key=lambda x: x[1].silhouette_score)
            f.write(
                f"\nRecommended Algorithm: {best_algo[0]} (Highest Silhouette Score)\n"
            )

            # Cluster Profiles
            for algo, profiles in profiles_dict.items():
                f.write(f"\n\n{algo.upper()} - CLUSTER PROFILES\n")
                f.write("=" * 80 + "\n")

                for profile in profiles:
                    f.write(
                        f"\nCluster {profile.cluster_id}: {profile.size} customers ({profile.percentage:.1f}%)\n"
                    )
                    f.write(
                        f"  Age: {profile.avg_age:.1f} years ({profile.age_category.value})\n"
                    )
                    f.write(
                        f"  Income: BDT {profile.avg_income:,.0f} ({profile.income_category.value})\n"
                    )
                    f.write(
                        f"  Spending: {profile.avg_spending:.1f} ({profile.spending_category.value})\n"
                    )
                    f.write(f"  Gender: {profile.dominant_gender}\n")
                    f.write(f"  Marketing Strategy: {profile.marketing_strategy}\n")

                    if profile.product_recommendations:
                        f.write(f"\n  Top Product Recommendations:\n")
                        for rec in profile.product_recommendations[:3]:
                            f.write(f"    - {rec['product']}: {rec['reason']}\n")

            # Validation Results
            f.write("\n\nVALIDATION RESULTS\n")
            f.write("-" * 80 + "\n")
            for algo, result in results.items():
                if result.validation_ari is not None:
                    f.write(f"{algo}: ARI = {result.validation_ari:.4f}\n")
                else:
                    f.write(f"{algo}: No validation data available\n")

            # Footer
            f.write("\n" + "=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")

        self.logger.info(f"Summary report saved: {report_path}")

    def export_results_to_csv(
        self, df: pd.DataFrame, results: Dict[str, ClusteringResult]
    ) -> None:
        """
        Export clustering results to CSV files.

        Args:
            df: Original dataframe
            results: Dictionary of clustering results
        """
        self.logger.info("\nExporting results to CSV files...")

        for algo, result in results.items():
            df_export = df.copy()
            df_export[f"{algo}_Cluster"] = result.labels

            filename = f"{algo.lower()}_results.csv"
            output_path = Path(self.config.OUTPUT_DIR) / filename
            df_export.to_csv(output_path, index=False)

            self.logger.info(f"  {algo} results exported: {output_path}")


# MAIN PIPELINE ORCHESTRATOR


class SegmentationPipeline:
    """Main pipeline orchestrator for customer segmentation analysis."""

    def __init__(self, config: Optional[AnalysisConfig] = None):
        """
        Initialize the segmentation pipeline.

        Args:
            config: Analysis configuration (uses default if None)
        """
        self.config = config or AnalysisConfig()
        self.logger = AnalysisLogger.setup(self.config.OUTPUT_DIR)

        # Initialize components
        self.data_loader = DataLoader(self.config, self.logger)
        self.preprocessor = DataPreprocessor(self.config, self.logger)
        self.eda = ExploratoryDataAnalysis(self.config, self.logger)
        self.cluster_finder = OptimalClusterFinder(self.config, self.logger)
        self.bi_engine = BusinessIntelligenceEngine(self.config, self.logger)
        self.viz_engine = VisualizationEngine(self.config, self.logger)
        self.report_gen = ReportGenerator(self.config, self.logger)

        # Data containers
        self.df: Optional[pd.DataFrame] = None
        self.df_validation: Optional[pd.DataFrame] = None
        self.X_scaled: Optional[np.ndarray] = None
        self.optimal_k: Optional[int] = None
        self.clustering_engine: Optional[ClusteringEngine] = None

    def run(
        self, skip_eda: bool = False, custom_k: Optional[int] = None
    ) -> Dict[str, ClusteringResult]:
        """
        Execute complete segmentation pipeline.

        Args:
            skip_eda: Skip exploratory data analysis
            custom_k: Use custom k instead of optimal k determination

        Returns:
            Dictionary of clustering results
        """
        try:
            self.logger.info("\n" + "=" * 80)
            self.logger.info("CUSTOMER SEGMENTATION ANALYSIS PIPELINE")
            self.logger.info("=" * 80)
            self.logger.info(f"Configuration: {self.config.DATA_FILE}")
            self.logger.info(f"Output Directory: {self.config.OUTPUT_DIR}")

            # Step 1: Load data
            self.df, self.df_validation = self.data_loader.load()

            # Step 2: Preprocess data
            self.df, self.X_scaled = self.preprocessor.preprocess(self.df)

            # Step 3: Exploratory data analysis
            if not skip_eda:
                self.eda.analyze(self.df)

            # Step 4: Find optimal clusters
            if custom_k is None:
                self.optimal_k = self.cluster_finder.find_optimal_k(self.X_scaled)
            else:
                self.optimal_k = custom_k
                self.logger.info(f"\nUsing custom k value: {self.optimal_k}")

            # Step 5: Initialize clustering engine
            self.clustering_engine = ClusteringEngine(
                self.config, self.logger, self.X_scaled, self.df
            )

            # Step 6: Run clustering algorithms
            results = self._run_all_algorithms()

            # Step 7: Validate against true labels (if available)
            if self.df_validation is not None:
                self._validate_results(results)

            # Step 8: Generate business insights
            profiles_dict = self._generate_business_insights(results)

            # Step 9: Create visualizations
            self._create_visualizations(results)

            # Step 10: Generate reports
            self.report_gen.generate_summary_report(
                results, profiles_dict, self.optimal_k
            )
            self.report_gen.export_results_to_csv(self.df, results)

            # Final summary
            self._print_final_summary(results)

            self.logger.info("\n" + "=" * 80)
            self.logger.info("ANALYSIS COMPLETED SUCCESSFULLY")
            self.logger.info("=" * 80)

            return results

        except Exception as e:
            self.logger.error(f"\nPipeline execution failed: {str(e)}")
            raise

    def _run_all_algorithms(self) -> Dict[str, ClusteringResult]:
        """Run all clustering algorithms."""
        results = {}

        # K-Means
        try:
            results[ClusteringAlgorithm.KMEANS.value] = (
                self.clustering_engine.run_kmeans(self.optimal_k)
            )
        except Exception as e:
            self.logger.error(f"K-Means failed: {e}")

        # Hierarchical
        try:
            results[ClusteringAlgorithm.HIERARCHICAL.value] = (
                self.clustering_engine.run_hierarchical(self.optimal_k)
            )
        except Exception as e:
            self.logger.error(f"Hierarchical clustering failed: {e}")

        # DBSCAN
        try:
            results[ClusteringAlgorithm.DBSCAN.value] = (
                self.clustering_engine.run_dbscan()
            )
        except Exception as e:
            self.logger.error(f"DBSCAN failed: {e}")

        # GMM
        try:
            results[ClusteringAlgorithm.GMM.value] = self.clustering_engine.run_gmm(
                self.optimal_k
            )
        except Exception as e:
            self.logger.error(f"GMM failed: {e}")

        return results

    def _validate_results(self, results: Dict[str, ClusteringResult]) -> None:
        """Validate clustering results against true labels."""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("VALIDATION AGAINST TRUE LABELS")
        self.logger.info("=" * 60)

        for algo in results.keys():
            self.clustering_engine.validate_against_true_labels(
                self.df_validation, algo
            )

    def _generate_business_insights(
        self, results: Dict[str, ClusteringResult]
    ) -> Dict[str, List[ClusterProfile]]:
        """Generate business insights for all algorithms."""
        profiles_dict = {}

        for algo, result in results.items():
            profiles = self.bi_engine.generate_cluster_profiles(
                self.df, result.labels, algo
            )
            profiles_dict[algo] = profiles

        return profiles_dict

    def _create_visualizations(self, results: Dict[str, ClusteringResult]) -> None:
        """Create all visualizations."""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("GENERATING VISUALIZATIONS")
        self.logger.info("=" * 60)

        # Cluster plots for each algorithm
        for algo, result in results.items():
            self.viz_engine.plot_clusters(self.df, result.labels, algo)

        # Dendrogram for hierarchical
        if ClusteringAlgorithm.HIERARCHICAL.value in results:
            self.viz_engine.plot_dendrogram(self.X_scaled)

        # Algorithm comparison
        self.viz_engine.plot_algorithm_comparison(results)

    def _print_final_summary(self, results: Dict[str, ClusteringResult]) -> None:
        """Print final summary of results."""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("FINAL SUMMARY")
        self.logger.info("=" * 60)

        # Best algorithm by silhouette score
        best_algo = max(results.items(), key=lambda x: x[1].silhouette_score)
        self.logger.info(f"\nBest Performing Algorithm: {best_algo[0]}")
        self.logger.info(f"  Silhouette Score: {best_algo[1].silhouette_score:.4f}")
        self.logger.info(f"  Number of Clusters: {best_algo[1].n_clusters}")

        # Output summary
        self.logger.info(f"\nAll results saved to: {self.config.OUTPUT_DIR}/")
        self.logger.info(f"  - Analysis log: analysis.log")
        self.logger.info(f"  - Summary report: analysis_summary.txt")
        self.logger.info(f"  - Visualizations: *.png files")
        self.logger.info(f"  - CSV exports: *_results.csv files")


# MAIN EXECUTION


def main():
    """Main execution function."""
    try:
        # Create configuration
        config = AnalysisConfig()

        # Initialize and run pipeline
        pipeline = SegmentationPipeline(config)
        results = pipeline.run(skip_eda=False)

        # Optional: Print best algorithm details
        best_algo = max(results.items(), key=lambda x: x[1].silhouette_score)
        print(f"\n{'='*60}")
        print(f"RECOMMENDED ALGORITHM: {best_algo[0]}")
        print(f"{'='*60}")
        print(f"Silhouette Score: {best_algo[1].silhouette_score:.4f}")
        print(f"Davies-Bouldin Index: {best_algo[1].davies_bouldin_index:.4f}")
        print(f"Calinski-Harabasz Score: {best_algo[1].calinski_harabasz_score:.0f}")
        print(f"Number of Clusters: {best_algo[1].n_clusters}")

        if best_algo[1].validation_ari is not None:
            print(f"Validation ARI: {best_algo[1].validation_ari:.4f}")

        return results

    except Exception as e:
        print(f"\nError: {str(e)}")
        raise


if __name__ == "__main__":

    # Run the analysis
    results = main()

    print("\n" + "=" * 60)
    print("Analysis complete! Check the output directory for results.")
    print("=" * 60)
