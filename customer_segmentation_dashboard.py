"""
Customer Segmentation Interactive Dashboard
============================================
A production-ready Streamlit application for customer segmentation analysis
with ML clustering algorithms and business intelligence insights.

Features:
    - Multiple clustering algorithms (K-Means, Hierarchical, DBSCAN, GMM)
    - Interactive visualizations (2D, 3D, correlation matrices)
    - Business insights and product recommendations
    - Performance metrics and validation
    - Export functionality for reports and data

Technical Stack:
    - Streamlit: Web application framework
    - Scikit-learn: Machine learning algorithms
    - Plotly: Interactive visualizations
    - Pandas/NumPy: Data processing

📊 Data → 🔍 Clustering → 🛍️ Products → 📈 Plan

Author: Shadin
Version: 2.0.0
Date: November 2025
License: MIT
"""

# IMPORTS

import warnings
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


# CONFIGURATION AND CONSTANTS


@dataclass(frozen=True)
class AppConfig:
    """Application configuration constants."""

    # File paths
    DATA_FILE: str = "Synthetic_Mall_Customers_3000.csv"
    VALIDATION_FILE: str = "Synthetic_Mall_Customers_3000_with_labels.csv"

    # Clustering defaults
    DEFAULT_CLUSTERS: int = 5
    DEFAULT_EPS: float = 0.32
    DEFAULT_MIN_SAMPLES: int = 6

    # Parameter ranges
    CLUSTER_RANGE: Tuple[int, int] = (2, 10)
    EPS_RANGE: Tuple[float, float, float] = (0.1, 1.0, 0.01)
    MIN_SAMPLES_RANGE: Tuple[int, int] = (3, 20)

    # Visualization
    SAMPLE_SIZE_3D: int = 1000
    COLOR_SCHEME: List[str] = None  # Will use plotly default

    # Features
    FEATURE_COLUMNS: List[str] = None
    REQUIRED_COLUMNS: List[str] = None

    def __post_init__(self):
        """Initialize mutable default values."""
        if self.FEATURE_COLUMNS is None:
            object.__setattr__(
                self,
                "FEATURE_COLUMNS",
                ["Age", "Annual Income (BDT)", "Spending Score (1-100)"],
            )
        if self.REQUIRED_COLUMNS is None:
            object.__setattr__(
                self,
                "REQUIRED_COLUMNS",
                ["CustomerID", "Gender"] + self.FEATURE_COLUMNS,
            )
        if self.COLOR_SCHEME is None:
            object.__setattr__(self, "COLOR_SCHEME", px.colors.qualitative.Set1)


class ClusterAlgorithm(str, Enum):
    """Supported clustering algorithms."""

    KMEANS = "K-Means"
    HIERARCHICAL = "Hierarchical"
    DBSCAN = "DBSCAN"
    GAUSSIAN_MIXTURE = "Gaussian Mixture"


class SegmentPriority(str, Enum):
    """Business priority levels for customer segments."""

    HIGH = "🔴 High"
    MEDIUM = "🟡 Medium"
    LOW = "🟢 Low"


# STREAMLIT PAGE CONFIGURATION


def configure_page() -> None:
    """Configure Streamlit page settings and custom CSS."""
    st.set_page_config(
        page_title="Customer Segmentation Dashboard",
        page_icon="🛍️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
            .main-header {
                font-size: 2.5rem;
                color: #1f77b4;
                text-align: center;
                margin-bottom: 1rem;
                font-weight: bold;
            }
            .metric-card {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 1.2rem;
                border-radius: 12px;
                margin: 0.5rem;
                color: white;
                text-align: center;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }
            .cluster-box {
                background-color: #f8f9fa;
                padding: 1rem;
                border-radius: 10px;
                border-left: 4px solid #1f77b4;
                margin: 0.5rem 0;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .insight-box {
                background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
                padding: 1rem;
                border-radius: 10px;
                margin: 0.5rem 0;
                color: white;
            }
            .stProgress > div > div > div > div {
                background: linear-gradient(to right, #667eea, #764ba2);
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


# DATA LOADING AND PREPROCESSING


class DataLoader:
    """Handles data loading and validation with caching."""

    def __init__(self, config: AppConfig):
        self.config = config

    @staticmethod
    @st.cache_data(show_spinner=False)
    def load_csv_files(
        data_path: str,
        validation_path: str,
        required_cols: List[str],
        feature_cols: List[str],
    ) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        Load main and validation datasets from CSV files.

        Args:
            data_path: Path to main dataset
            validation_path: Path to validation dataset with true labels
            required_cols: List of required column names
            feature_cols: List of feature columns for clustering

        Returns:
            Tuple of (main_dataframe, validation_dataframe)
        """
        try:
            # Load main dataset
            df = pd.read_csv(data_path)

            # Validate columns
            missing_cols = set(required_cols) - set(df.columns)
            if missing_cols:
                st.error(f"❌ Missing required columns: {missing_cols}")
                return None, None

            # Validate data types
            for col in feature_cols:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    st.error(
                        f"❌ Column '{col}' must be numeric, found: {df[col].dtype}"
                    )
                    return None, None

            # Handle missing values
            if df[feature_cols].isnull().any().any():
                null_counts = df[feature_cols].isnull().sum()
                st.warning(f"⚠️ Found missing values:\n{null_counts[null_counts > 0]}")
                df = df.dropna(subset=feature_cols)
                st.info(f"✓ Removed rows with missing values. Remaining: {len(df):,}")

            # Handle infinite values
            if np.isinf(df[feature_cols]).any().any():
                st.warning("⚠️ Infinite values detected and removed")
                df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=feature_cols)

            # Load validation data if available
            df_validation = None
            if Path(validation_path).exists():
                try:
                    df_validation = pd.read_csv(validation_path)
                    st.sidebar.success("✅ Validation data loaded")
                except Exception as e:
                    st.sidebar.warning(f"⚠️ Could not load validation data: {e}")

            return df, df_validation

        except FileNotFoundError:
            st.error(f"❌ File not found: {data_path}")
            st.info("Please ensure the CSV file exists in the current directory")
            return None, None
        except Exception as e:
            st.error(f"❌ Error loading data: {str(e)}")
            return None, None

    def load(self) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """Load datasets using cached function."""
        return self.load_csv_files(
            self.config.DATA_FILE,
            self.config.VALIDATION_FILE,
            self.config.REQUIRED_COLUMNS,
            self.config.FEATURE_COLUMNS,
        )


class DataPreprocessor:
    """Handles data preprocessing and scaling."""

    @staticmethod
    @st.cache_data(show_spinner=False)
    def scale_features(
        df: pd.DataFrame, feature_cols: List[str]
    ) -> Tuple[np.ndarray, StandardScaler]:
        """
        Scale features using StandardScaler.

        Args:
            df: Input dataframe
            feature_cols: List of columns to scale

        Returns:
            Tuple of (scaled_features, fitted_scaler)
        """
        X = df[feature_cols].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return X_scaled, scaler


# CLUSTERING ENGINE


@dataclass
class ClusteringResult:
    """Container for clustering results."""

    labels: np.ndarray
    n_clusters: int
    silhouette_score: float
    davies_bouldin_index: float
    calinski_harabasz_score: float
    adjusted_rand_index: Optional[float] = None
    warning_message: Optional[str] = None

    @property
    def is_valid(self) -> bool:
        """Check if clustering produced valid results."""
        return self.labels is not None and self.n_clusters > 1


class ClusteringEngine:
    """Handles all clustering operations with caching."""

    def __init__(self, config: AppConfig):
        self.config = config

    @staticmethod
    @st.cache_data(show_spinner=False, hash_funcs={pd.Series: lambda x: id(x)})
    def perform_clustering(
        X_scaled: np.ndarray,
        algorithm: str,
        n_clusters: int,
        eps: float,
        min_samples: int,
        validation_labels: Optional[pd.Series] = None,
    ) -> ClusteringResult:
        """
        Perform clustering with specified algorithm and parameters.

        Args:
            X_scaled: Scaled feature matrix
            algorithm: Name of clustering algorithm
            n_clusters: Number of clusters (for K-Means, Hierarchical, GMM)
            eps: DBSCAN epsilon parameter
            min_samples: DBSCAN min_samples parameter
            validation_labels: True labels for validation (optional)

        Returns:
            ClusteringResult object with metrics and labels
        """
        try:
            # Select and fit model
            if algorithm == ClusterAlgorithm.KMEANS.value:
                model = KMeans(
                    n_clusters=n_clusters, random_state=42, n_init=10, max_iter=300
                )
                labels = model.fit_predict(X_scaled)

            elif algorithm == ClusterAlgorithm.HIERARCHICAL.value:
                model = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
                labels = model.fit_predict(X_scaled)

            elif algorithm == ClusterAlgorithm.DBSCAN.value:
                model = DBSCAN(eps=eps, min_samples=min_samples)
                labels = model.fit_predict(X_scaled)

            elif algorithm == ClusterAlgorithm.GAUSSIAN_MIXTURE.value:
                model = GaussianMixture(
                    n_components=n_clusters, random_state=42, max_iter=200
                )
                labels = model.fit_predict(X_scaled)
            else:
                return ClusteringResult(
                    labels=None,
                    n_clusters=0,
                    silhouette_score=-1,
                    davies_bouldin_index=float("inf"),
                    calinski_harabasz_score=0,
                    warning_message=f"Unknown algorithm: {algorithm}",
                )

            # Calculate number of clusters
            n_clusters_found = len(set(labels)) - (1 if -1 in labels else 0)

            # Handle edge cases
            if n_clusters_found == 0:
                return ClusteringResult(
                    labels=labels,
                    n_clusters=0,
                    silhouette_score=-1,
                    davies_bouldin_index=float("inf"),
                    calinski_harabasz_score=0,
                    warning_message=f"⚠️ {algorithm} found no valid clusters. Try adjusting parameters.",
                )

            if n_clusters_found == 1:
                return ClusteringResult(
                    labels=labels,
                    n_clusters=1,
                    silhouette_score=-1,
                    davies_bouldin_index=float("inf"),
                    calinski_harabasz_score=0,
                    warning_message=f"⚠️ {algorithm} found only 1 cluster. Consider different parameters.",
                )

            # Calculate metrics - use full dataset (including noise for DBSCAN)
            try:
                sil_score = silhouette_score(X_scaled, labels)
                db_index = davies_bouldin_score(X_scaled, labels)
                ch_score = calinski_harabasz_score(X_scaled, labels)
            except Exception as metric_error:
                # Fallback: filter noise points if metrics calculation fails
                if -1 in labels:
                    mask = labels != -1
                    if np.sum(mask) > 1:
                        sil_score = silhouette_score(X_scaled[mask], labels[mask])
                        db_index = davies_bouldin_score(X_scaled[mask], labels[mask])
                        ch_score = calinski_harabasz_score(X_scaled[mask], labels[mask])
                    else:
                        sil_score, db_index, ch_score = -1, float("inf"), 0
                else:
                    raise metric_error

            # Calculate validation metric if available
            ari = None
            if validation_labels is not None and n_clusters_found > 1:
                try:
                    unique_labels = {
                        label: idx
                        for idx, label in enumerate(validation_labels.unique())
                    }
                    true_labels_numeric = validation_labels.map(unique_labels)
                    ari = adjusted_rand_score(true_labels_numeric, labels)
                except Exception:
                    pass

            return ClusteringResult(
                labels=labels,
                n_clusters=n_clusters_found,
                silhouette_score=sil_score,
                davies_bouldin_index=db_index,
                calinski_harabasz_score=ch_score,
                adjusted_rand_index=ari,
                warning_message=None,
            )

        except Exception as e:
            return ClusteringResult(
                labels=None,
                n_clusters=0,
                silhouette_score=-1,
                davies_bouldin_index=float("inf"),
                calinski_harabasz_score=0,
                warning_message=f"❌ Clustering error in {algorithm}: {str(e)}",
            )

    def cluster(
        self,
        X_scaled: np.ndarray,
        algorithm: str,
        n_clusters: int,
        eps: float,
        min_samples: int,
        validation_labels: Optional[pd.Series] = None,
    ) -> ClusteringResult:
        """Wrapper for cached clustering function."""
        return self.perform_clustering(
            X_scaled, algorithm, n_clusters, eps, min_samples, validation_labels
        )


# VISUALIZATION ENGINE


class VisualizationEngine:
    """Handles all visualization creation with caching."""

    def __init__(self, config: AppConfig):
        self.config = config

    @staticmethod
    @st.cache_data(show_spinner=False)
    def create_cluster_subplots(
        df: pd.DataFrame,
        labels: np.ndarray,
        algorithm_name: str,
        color_scheme: List[str],
    ) -> go.Figure:
        """
        Create 2x2 subplot figure with cluster visualizations.

        Args:
            df: DataFrame with customer data
            labels: Cluster labels
            algorithm_name: Name of clustering algorithm
            color_scheme: Color palette for clusters

        Returns:
            Plotly Figure object
        """
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Income vs Spending Score",
                "Age vs Spending Score",
                "Age vs Income",
                "Cluster Distribution",
            ),
            horizontal_spacing=0.2,
            vertical_spacing=0.20,
        )

        unique_clusters = sorted(set(labels))

        def add_scatter(
            x_col: str, y_col: str, row: int, col: int, show_legend: bool = False
        ):
            """Helper to add scatter traces."""
            for i, cluster in enumerate(unique_clusters):
                cluster_name = "Noise" if cluster == -1 else f"Cluster {cluster}"
                color = "gray" if cluster == -1 else color_scheme[i % len(color_scheme)]

                cluster_mask = labels == cluster
                cluster_size = np.sum(cluster_mask)

                fig.add_trace(
                    go.Scatter(
                        x=df[cluster_mask][x_col],
                        y=df[cluster_mask][y_col],
                        mode="markers",
                        name=(
                            f"{cluster_name} (n={cluster_size})"
                            if show_legend
                            else cluster_name
                        ),
                        marker=dict(size=6, opacity=0.7, color=color),
                        legendgroup=f"cluster_{cluster}",
                        showlegend=show_legend,
                    ),
                    row=row,
                    col=col,
                )

        # Add all scatter plots
        add_scatter(
            "Annual Income (BDT)", "Spending Score (1-100)", 1, 1, show_legend=True
        )
        add_scatter("Age", "Spending Score (1-100)", 1, 2)
        add_scatter("Age", "Annual Income (BDT)", 2, 1)

        # Cluster distribution bar chart
        cluster_counts = pd.Series(labels).value_counts().sort_index()
        cluster_names = [
            "Noise" if c == -1 else f"Cluster {c}" for c in cluster_counts.index
        ]

        fig.add_trace(
            go.Bar(
                x=cluster_names,
                y=cluster_counts.values,
                name="Cluster Size",
                marker_color=[
                    (
                        "gray"
                        if cluster_counts.index[i] == -1
                        else color_scheme[i % len(color_scheme)]
                    )
                    for i in range(len(cluster_counts))
                ],
                showlegend=False,
            ),
            row=2,
            col=2,
        )

        # Update layout
        fig.update_layout(
            height=700,
            title_text=f"{algorithm_name} Clustering - {len(df):,} Customers",
            title_x=0.5,
            showlegend=True,
        )

        # Update axes
        fig.update_xaxes(title_text="Annual Income (BDT)", row=1, col=1)
        fig.update_yaxes(title_text="Spending Score", row=1, col=1)
        fig.update_xaxes(title_text="Age", row=1, col=2)
        fig.update_yaxes(title_text="Spending Score", row=1, col=2)
        fig.update_xaxes(title_text="Age", row=2, col=1)
        fig.update_yaxes(title_text="Annual Income (BDT)", row=2, col=1)
        fig.update_xaxes(title_text="Cluster", row=2, col=2)
        fig.update_yaxes(title_text="Number of Customers", row=2, col=2)

        return fig

    @staticmethod
    @st.cache_data(show_spinner=False)
    def create_3d_scatter(
        df: pd.DataFrame,
        labels: np.ndarray,
        algorithm_name: str,
        sample_size: int,
        color_scheme: List[str],
    ) -> Tuple[go.Figure, bool]:
        """
        Create 3D scatter plot with optional sampling.

        Args:
            df: DataFrame with customer data
            labels: Cluster labels
            algorithm_name: Name of clustering algorithm
            sample_size: Maximum number of points to display
            color_scheme: Color palette

        Returns:
            Tuple of (Figure, was_sampled)
        """
        # Sample if necessary
        sampled = len(df) > sample_size
        if sampled:
            sample_idx = np.random.choice(len(df), sample_size, replace=False)
            df_plot = df.iloc[sample_idx].copy()
            labels_plot = labels[sample_idx]
        else:
            df_plot = df
            labels_plot = labels

        # Create figure
        fig = px.scatter_3d(
            df_plot,
            x="Annual Income (BDT)",
            y="Spending Score (1-100)",
            z="Age",
            color=labels_plot.astype(str),
            title=f"{algorithm_name} - 3D Visualization"
            + (f" (Sample: {sample_size})" if sampled else ""),
            opacity=0.7,
            color_discrete_sequence=color_scheme,
            labels={"color": "Cluster"},
        )

        fig.update_layout(height=600)

        return fig, sampled

    def create_cluster_visualizations(
        self, df: pd.DataFrame, labels: np.ndarray, algorithm_name: str
    ) -> go.Figure:
        """Create cluster visualization subplots."""
        return self.create_cluster_subplots(
            df, labels, algorithm_name, self.config.COLOR_SCHEME
        )

    def create_3d_visualization(
        self, df: pd.DataFrame, labels: np.ndarray, algorithm_name: str
    ) -> Tuple[go.Figure, bool]:
        """Create 3D visualization with sampling."""
        return self.create_3d_scatter(
            df,
            labels,
            algorithm_name,
            self.config.SAMPLE_SIZE_3D,
            self.config.COLOR_SCHEME,
        )


# BUSINESS INTELLIGENCE ENGINE


@dataclass
class ClusterInsight:
    """Business insights for a customer cluster."""

    cluster_id: int
    size: int
    percentage: float
    avg_age: float
    avg_income: float
    avg_spending: float
    dominant_gender: str
    age_group: str
    income_group: str
    spending_group: str
    recommendation: str
    priority: str
    spending_category: str = ""


class BusinessIntelligence:
    """Generates business insights and recommendations."""

    @staticmethod
    def categorize_age(age: float) -> str:
        """Categorize age into groups."""
        if age < 30:
            return "Young"
        elif age < 50:
            return "Middle-aged"
        else:
            return "Senior"

    @staticmethod
    def categorize_income(income: float) -> str:
        """Categorize income into groups."""
        if income < 500000:
            return "Low"
        elif income < 1000000:
            return "Medium"
        else:
            return "High"

    @staticmethod
    def categorize_spending(spending: float) -> str:
        """Categorize spending into groups."""
        if spending < 40:
            return "Low"
        elif spending < 70:
            return "Medium"
        else:
            return "High"

    @staticmethod
    def generate_recommendation(
        income_group: str, spending_group: str
    ) -> Tuple[str, str]:
        """
        Generate business recommendation and priority.

        Returns:
            Tuple of (recommendation, priority)
        """
        if spending_group == "High" and income_group == "High":
            return (
                "Premium products, VIP programs, exclusive offers, personalized service",
                SegmentPriority.HIGH.value,
            )
        elif spending_group == "High" and income_group in ["Medium", "Low"]:
            return (
                "Value bundles, loyalty rewards, quality products at competitive prices",
                SegmentPriority.MEDIUM.value,
            )
        elif spending_group == "Low" and income_group == "High":
            return (
                "Trust-building campaigns, product demonstrations, value proposition focus",
                SegmentPriority.MEDIUM.value,
            )
        else:
            return (
                "Standard campaigns, seasonal promotions, volume discounts",
                SegmentPriority.LOW.value,
            )

    @staticmethod
    @st.cache_data(show_spinner=False)
    def analyze_clusters(df: pd.DataFrame, labels: np.ndarray) -> List[ClusterInsight]:
        """
        Generate comprehensive business insights for each cluster.

        Args:
            df: DataFrame with customer data
            labels: Cluster labels

        Returns:
            List of ClusterInsight objects
        """
        insights = []

        for cluster_id in sorted(set(labels)):
            if cluster_id == -1:  # Skip noise
                continue

            cluster_mask = labels == cluster_id
            cluster_data = df[cluster_mask]

            # Calculate statistics
            size = np.sum(cluster_mask)
            percentage = (size / len(df)) * 100
            avg_age = cluster_data["Age"].mean()
            avg_income = cluster_data["Annual Income (BDT)"].mean()
            avg_spending = cluster_data["Spending Score (1-100)"].mean()
            dominant_gender = (
                cluster_data["Gender"].mode()[0] if len(cluster_data) > 0 else "Unknown"
            )

            # Categorize
            age_group = BusinessIntelligence.categorize_age(avg_age)
            income_group = BusinessIntelligence.categorize_income(avg_income)
            spending_group = BusinessIntelligence.categorize_spending(avg_spending)

            # Generate recommendation
            recommendation, priority = BusinessIntelligence.generate_recommendation(
                income_group, spending_group
            )

            insights.append(
                ClusterInsight(
                    cluster_id=cluster_id,
                    size=size,
                    percentage=percentage,
                    avg_age=avg_age,
                    avg_income=avg_income,
                    avg_spending=avg_spending,
                    dominant_gender=dominant_gender,
                    age_group=age_group,
                    income_group=income_group,
                    spending_group=spending_group,
                    recommendation=recommendation,
                    priority=priority,
                )
            )

        return insights

    def get_insights(
        self, df: pd.DataFrame, labels: np.ndarray
    ) -> List[ClusterInsight]:
        """Get cluster insights using cached analysis."""
        return self.analyze_clusters(df, labels)


# PRODUCT RECOMMENDATION ENGINE


@dataclass
class ProductRecommendation:
    """Product recommendation with business rationale."""

    product: str
    price: str
    conversion: str
    priority: str
    reason: str
    strategy: str


@dataclass
class DemographicInsight:
    """Demographic insights for targeting."""

    demographic: str
    behavior: str
    approach: str


class ProductRecommendationEngine:
    """Generates product recommendations based on cluster characteristics."""

    @staticmethod
    def get_demographic_insight(age_group: str) -> DemographicInsight:
        """Get demographic insights based on age group."""
        insights = {
            "Young": DemographicInsight(
                demographic="Young Adults (18-30 years)",
                behavior="Tech-savvy, trend-conscious, social media active, value experiences",
                approach="Digital-first marketing, influencer partnerships, mobile campaigns",
            ),
            "Middle-aged": DemographicInsight(
                demographic="Middle-aged Adults (30-50 years)",
                behavior="Career-focused, family-oriented, value quality and reliability",
                approach="Multi-channel marketing, emphasize value and quality, email campaigns",
            ),
            "Senior": DemographicInsight(
                demographic="Senior Adults (50+ years)",
                behavior="Experience-rich, budget-conscious, prefer personal service",
                approach="Traditional marketing mix, customer service focus, loyalty programs",
            ),
        }
        return insights.get(age_group, insights["Middle-aged"])

    @staticmethod
    def generate_recommendations(
        age_group: str, income_group: str, spending_group: str, avg_age: float
    ) -> Tuple[List[ProductRecommendation], DemographicInsight]:
        """
        Generate product recommendations for a cluster.

        Args:
            age_group: Age category
            income_group: Income category
            spending_group: Spending category
            avg_age: Average age

        Returns:
            Tuple of (recommendations_list, demographic_insight)
        """
        # Get demographic insights
        demo_insight = ProductRecommendationEngine.get_demographic_insight(age_group)

        recommendations = []

        # High Income + High Spending
        if income_group == "High" and spending_group == "High":
            recommendations = [
                ProductRecommendation(
                    product="📱 iPhone Pro Max",
                    price="BDT 140,000 - 180,000",
                    conversion="High (75-85%)",
                    priority="🔴 Primary Target",
                    reason="High purchasing power with willingness to spend on premium products.",
                    strategy="VIP early access, exclusive events, premium service packages",
                ),
                ProductRecommendation(
                    product="💻 MacBook Pro",
                    price="BDT 180,000 - 280,000",
                    conversion="High (70-80%)",
                    priority="🔴 Primary Target",
                    reason="Professional segment seeking premium productivity tools.",
                    strategy="B2B partnerships, corporate bundles, professional support",
                ),
            ]

        # High Income + Low Spending
        elif income_group == "High" and spending_group in ["Low", "Medium"]:
            recommendations = [
                ProductRecommendation(
                    product="🖥️ HP Business Laptop",
                    price="BDT 65,000 - 95,000",
                    conversion="High (70-80%)",
                    priority="🔴 Primary Target",
                    reason="Value-conscious despite high income. Seeks quality at reasonable prices.",
                    strategy="Emphasize ROI, durability, warranty, customer reviews",
                ),
            ]

        # Medium/Low Income + High Spending
        elif income_group in ["Low", "Medium"] and spending_group == "High":
            recommendations = [
                ProductRecommendation(
                    product="🖥️ HP Mid-Range Laptop",
                    price="BDT 45,000 - 70,000",
                    conversion="High (75-85%)",
                    priority="🔴 Primary Target",
                    reason="Aspirational but affordable. EMI options make it accessible.",
                    strategy="Flexible payment plans, 0% EMI, student discounts",
                ),
                ProductRecommendation(
                    product="🎧 Wireless Headphones",
                    price="BDT 3,000 - 8,000",
                    conversion="High (70-80%)",
                    priority="🔴 Primary Target",
                    reason="Lifestyle accessory within budget with perceived value.",
                    strategy="Social media campaigns, influencer marketing, cashback offers",
                ),
            ]

        # Default recommendations
        else:
            recommendations = [
                ProductRecommendation(
                    product="🎧 Budget Headphones",
                    price="BDT 1,500 - 3,500",
                    conversion="High (75-85%)",
                    priority="🔴 Primary Target",
                    reason="Essential accessory at accessible price point.",
                    strategy="Affordability focus, quality assurance, first-buyer incentives",
                ),
                ProductRecommendation(
                    product="🔊 Basic Bluetooth Speaker",
                    price="BDT 1,800 - 3,500",
                    conversion="Medium (65-75%)",
                    priority="🟡 Secondary Target",
                    reason="Entry-level lifestyle product with volume sales potential.",
                    strategy="Volume discounts, festival bonanzas, flash sales",
                ),
            ]

        return recommendations, demo_insight


# MAIN DASHBOARD APPLICATION


class CustomerSegmentationDashboard:
    """Main dashboard orchestrator."""

    def __init__(self):
        self.config = AppConfig()
        self.data_loader = DataLoader(self.config)
        self.preprocessor = DataPreprocessor()
        self.clustering_engine = ClusteringEngine(self.config)
        self.viz_engine = VisualizationEngine(self.config)
        self.bi_engine = BusinessIntelligence()
        self.product_engine = ProductRecommendationEngine()

        # Data containers
        self.df: Optional[pd.DataFrame] = None
        self.df_validation: Optional[pd.DataFrame] = None
        self.X_scaled: Optional[np.ndarray] = None
        self.scaler: Optional[StandardScaler] = None

    def load_data(self) -> bool:
        """Load and preprocess data."""
        with st.spinner("📂 Loading data..."):
            self.df, self.df_validation = self.data_loader.load()

        if self.df is None:
            return False

        # Display dataset info in sidebar
        self._display_data_info()

        # Preprocess data
        self.X_scaled, self.scaler = self.preprocessor.scale_features(
            self.df, self.config.FEATURE_COLUMNS
        )

        return True

    def _display_data_info(self) -> None:
        """Display dataset information in sidebar."""
        st.sidebar.markdown("### 📊 Dataset Info")
        st.sidebar.write(f"**Total Customers:** {len(self.df):,}")
        st.sidebar.write(f"**Features:** {len(self.df.columns)}")

        gender_counts = self.df["Gender"].value_counts()
        st.sidebar.write(f"**Male:** {gender_counts.get('Male', 0):,}")
        st.sidebar.write(f"**Female:** {gender_counts.get('Female', 0):,}")

        if self.df_validation is not None:
            st.sidebar.info("ℹ️ Validation data available")

    def run_clustering(
        self, algorithm: str, n_clusters: int, eps: float, min_samples: int
    ) -> ClusteringResult:
        """Execute clustering analysis."""
        validation_labels = None
        if (
            self.df_validation is not None
            and "True_Segment" in self.df_validation.columns
        ):
            validation_labels = self.df_validation["True_Segment"]

        return self.clustering_engine.cluster(
            self.X_scaled, algorithm, n_clusters, eps, min_samples, validation_labels
        )

    def render(self) -> None:
        """Render the complete dashboard."""
        # Header
        st.markdown(
            '<h1 style="text-align: center; color: #1f77b4;">'
            "🛍️ Customer Segmentation Dashboard</h1>",
            unsafe_allow_html=True,
        )
        st.markdown(
            '<h3 style="text-align: center;">' "ML-Powered Customer Analytics</h3>",
            unsafe_allow_html=True,
        )

        # Load data
        if not self.load_data():
            st.stop()

        # Sidebar controls
        self._render_sidebar()

        # Main content tabs
        self._render_tabs()

        # Footer
        self._render_footer()

    def _render_sidebar(self) -> None:
        """Render sidebar controls."""
        st.sidebar.markdown("---")
        st.sidebar.markdown("## ⚙️ Algorithm Settings")

        # Algorithm selection
        algorithm = st.sidebar.selectbox(
            "Select Clustering Algorithm",
            [algo.value for algo in ClusterAlgorithm],
            help="Choose the clustering algorithm to analyze customer data",
        )

        # Store in session state
        st.session_state.setdefault("algorithm", algorithm)
        st.session_state["algorithm"] = algorithm

        # Algorithm-specific parameters
        if algorithm in [
            ClusterAlgorithm.KMEANS.value,
            ClusterAlgorithm.HIERARCHICAL.value,
            ClusterAlgorithm.GAUSSIAN_MIXTURE.value,
        ]:
            n_clusters = st.sidebar.slider(
                "Number of Clusters",
                min_value=self.config.CLUSTER_RANGE[0],
                max_value=self.config.CLUSTER_RANGE[1],
                value=self.config.DEFAULT_CLUSTERS,
            )
            eps = self.config.DEFAULT_EPS
            min_samples = self.config.DEFAULT_MIN_SAMPLES
        else:  # DBSCAN
            eps = st.sidebar.slider(
                "EPS (Neighborhood radius)",
                min_value=self.config.EPS_RANGE[0],
                max_value=self.config.EPS_RANGE[1],
                value=self.config.DEFAULT_EPS,
                step=self.config.EPS_RANGE[2],
            )
            min_samples = st.sidebar.slider(
                "Min Samples",
                min_value=self.config.MIN_SAMPLES_RANGE[0],
                max_value=self.config.MIN_SAMPLES_RANGE[1],
                value=self.config.DEFAULT_MIN_SAMPLES,
            )
            n_clusters = self.config.DEFAULT_CLUSTERS

        # Store parameters
        st.session_state.update(
            {"n_clusters": n_clusters, "eps": eps, "min_samples": min_samples}
        )

    def _render_tabs(self) -> None:
        """Render main content tabs."""
        tabs = st.tabs(
            [
                "📊 Data Overview",
                "🔍 Clustering Analysis",
                "📦 Product Recommendations",
                "📈 Action Plan",
            ]
        )

        with tabs[0]:
            self._render_data_overview_tab()

        with tabs[1]:
            self._render_clustering_tab()

        with tabs[2]:
            self._render_recommendations_tab()

        with tabs[3]:
            self._render_action_plan_tab()

    def _render_data_overview_tab(self) -> None:
        """Render data overview tab."""
        st.header(f"Dataset Overview - {len(self.df):,} Customers")

        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Customers", f"{len(self.df):,}")
        with col2:
            st.metric("Average Age", f"{self.df['Age'].mean():.1f} years")
        with col3:
            st.metric(
                "Average Income", f"BDT {self.df['Annual Income (BDT)'].mean():,.0f}"
            )
        with col4:
            st.metric(
                "Avg Spending Score", f"{self.df['Spending Score (1-100)'].mean():.1f}"
            )

        # Data exploration
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("📋 Data Sample")
            st.dataframe(self.df.head(10), use_container_width=True)

            st.subheader("📊 Feature Distributions")
            selected_feature = st.selectbox(
                "Select feature", self.config.FEATURE_COLUMNS
            )
            fig = px.histogram(
                self.df,
                x=selected_feature,
                title=f"Distribution of {selected_feature}",
                color_discrete_sequence=["#1f77b4"],
                nbins=30,
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("📈 Quick Statistics")
            st.dataframe(self.df[self.config.FEATURE_COLUMNS].describe())

            st.subheader("👥 Gender Distribution")
            fig = px.pie(
                self.df,
                names="Gender",
                title="Customer Gender Distribution",
                color_discrete_sequence=["#FD9FAD", "#67D3FD"],
            )
            st.plotly_chart(fig, use_container_width=True)

        # Correlation heatmap
        st.subheader("🔥 Feature Correlation")
        corr_matrix = self.df[self.config.FEATURE_COLUMNS].corr()
        fig = px.imshow(
            corr_matrix,
            text_auto=".3f",
            aspect="auto",
            color_continuous_scale="RdBu_r",
            title="Feature Correlation Heatmap",
        )
        st.plotly_chart(fig, use_container_width=True)

    def _render_clustering_tab(self) -> None:
        """Render clustering analysis tab."""
        st.header("🔍 Clustering Analysis")

        if st.button(
            "🎯 Run Clustering Analysis", type="primary", use_container_width=True
        ):
            self._execute_clustering_analysis()
        else:
            self._display_cached_results()

    def _execute_clustering_analysis(self) -> None:
        """Execute and display clustering analysis."""
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            # Get parameters from session state
            algorithm = st.session_state.get("algorithm")
            n_clusters = st.session_state.get("n_clusters")
            eps = st.session_state.get("eps")
            min_samples = st.session_state.get("min_samples")

            # Run clustering
            status_text.text(f"🔄 Running {algorithm} clustering...")
            progress_bar.progress(20)

            result = self.run_clustering(algorithm, n_clusters, eps, min_samples)
            progress_bar.progress(40)

            if result.labels is None:
                st.error("❌ Clustering failed. Please try different parameters.")
                return

            if result.warning_message:
                st.warning(result.warning_message)

            # Generate visualizations
            status_text.text("📊 Creating visualizations...")
            progress_bar.progress(60)

            fig_clusters = self.viz_engine.create_cluster_visualizations(
                self.df, result.labels, algorithm
            )
            progress_bar.progress(70)

            fig_3d, sampled = self.viz_engine.create_3d_visualization(
                self.df, result.labels, algorithm
            )
            progress_bar.progress(80)

            # Generate insights
            status_text.text("💡 Generating insights...")
            insights = self.bi_engine.get_insights(self.df, result.labels)
            progress_bar.progress(100)

            # Store results
            st.session_state.update(
                {
                    "clustering_result": result,
                    "insights": insights,
                    "fig_clusters": fig_clusters,
                    "fig_3d": fig_3d,
                    "sampled_3d": sampled,
                }
            )

            progress_bar.empty()
            status_text.empty()
            st.success(f"✅ {algorithm} clustering completed!")

            # Display results
            self._display_clustering_results(result, fig_clusters, fig_3d, sampled)

        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"❌ Unexpected error: {str(e)}")

    def _display_clustering_results(
        self,
        result: ClusteringResult,
        fig_clusters: go.Figure,
        fig_3d: go.Figure,
        sampled: bool,
    ) -> None:
        """Display clustering results."""
        # Metrics
        st.markdown("### 📊 Performance Metrics")
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric("Clusters Found", result.n_clusters)
        with col2:
            st.metric("Silhouette Score", f"{result.silhouette_score:.3f}")
        with col3:
            st.metric("Davies-Bouldin", f"{result.davies_bouldin_index:.3f}")
        with col4:
            st.metric("Calinski-Harabasz", f"{result.calinski_harabasz_score:.0f}")
        with col5:
            ari_text = (
                f"{result.adjusted_rand_index:.3f}"
                if result.adjusted_rand_index
                else "N/A"
            )
            st.metric("Validation ARI", ari_text)

        # Visualizations
        st.markdown("### 📈 Cluster Visualization")
        st.plotly_chart(fig_clusters, use_container_width=True)

        st.markdown("### 🌐 3D Cluster View")
        if sampled:
            st.info(
                f"📊 Showing {self.config.SAMPLE_SIZE_3D} of {len(self.df):,} customers"
            )
        st.plotly_chart(fig_3d, use_container_width=True)

    def _display_cached_results(self) -> None:
        """Display previously computed clustering results."""
        if "clustering_result" in st.session_state:
            st.info("💡 Previously computed results shown. Click button to recompute.")
            result = st.session_state["clustering_result"]
            self._display_clustering_results(
                result,
                st.session_state["fig_clusters"],
                st.session_state["fig_3d"],
                st.session_state["sampled_3d"],
            )
        else:
            st.info("👆 Click the button above to run clustering analysis")

    def _render_recommendations_tab(self) -> None:
        """Render product recommendations tab."""
        st.header("📦 Product Recommendations by Cluster")

        if "insights" not in st.session_state or not st.session_state["insights"]:
            st.info("📊 Run clustering analysis to see product recommendations")
            return

        insights = st.session_state["insights"]

        for insight in insights:
            # ADD SPENDING SCORE DISPLAY
            spending_display = f"{insight.avg_spending:.1f}/100"

            with st.expander(
                f"🎯 Cluster {insight.cluster_id} - {insight.size:,} customers - "
                f"{insight.age_group}, {insight.income_group} Income | "
                f"💰 Spending: {spending_display}",  # ← NEW
                expanded=True,
            ):
                # OPTIONAL: Add metric box at top
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Cluster Size", f"{insight.size:,}")
                with col2:
                    st.metric("Avg Age", f"{insight.avg_age:.1f}y")
                with col3:
                    st.metric("Avg Income", f"BDT {insight.avg_income:,.0f}")
                with col4:
                    st.metric("Spending Score", f"{insight.avg_spending:.1f}")

                st.markdown("---")

                # Demographic insights
                recs, demo_insight = self.product_engine.generate_recommendations(
                    insight.age_group,
                    insight.income_group,
                    insight.spending_group,
                    insight.avg_age,
                )

                st.info(
                    f"""
                    **{demo_insight.demographic}**

                    **Behavior**: {demo_insight.behavior}

                    **Approach**: {demo_insight.approach}
                    """
                )

                st.markdown("---")
                st.markdown("#### 📦 Recommended Products")

                # Display recommendations
                for rec in recs:
                    col1, col2 = st.columns([1, 2])

                    with col1:
                        st.markdown(
                            f"""
                        ### {rec.product}
                        **{rec.priority}**

                        💰 **Price**: {rec.price}
                        📈 **Conversion**: {rec.conversion}
                        """
                        )

                    with col2:
                        st.markdown(
                            f"""
                        **🎯 Why Recommend:**
                        {rec.reason}

                        **📋 Marketing Strategy:**
                        {rec.strategy}
                        """
                        )

                    if "Primary" in rec.priority:
                        st.success("✅ High Priority - Focus marketing efforts here")

                    st.markdown("---")

    def _render_action_plan_tab(self) -> None:
        """Render action plan tab."""
        st.header("📈 Marketing Action Plan")

        if "insights" not in st.session_state or not st.session_state["insights"]:
            st.info("📊 Run clustering analysis to generate action plans")
            return

        insights = st.session_state["insights"]

        st.markdown("### 🎯 Action Plan by Priority Level")

        # Display ALL clusters organized by priority
        st.markdown("---")

        # 🔴 HIGH PRIORITY CLUSTERS
        st.markdown("#### 🔴 HIGH PRIORITY ACTIONS")
        high_priority = [i for i in insights if "High" in i.priority]

        if high_priority:
            for cluster in high_priority:
                with st.expander(
                    f"Cluster {cluster.cluster_id} - {cluster.size:,} customers ({cluster.percentage:.1f}%)",
                    expanded=True,
                ):
                    col1, col2 = st.columns([2, 1])

                    with col1:
                        st.markdown(
                            f"""
                        **Segment Profile:**
                        - Age: {cluster.avg_age:.1f} years ({cluster.age_group})
                        - Income: BDT {cluster.avg_income:,.0f} ({cluster.income_group})
                        - Spending: {cluster.avg_spending:.1f}/100 ({cluster.spending_group})

                        **Strategy:**
                        {cluster.recommendation}

                        **Expected Impact:** High
                        **Timeline:** 3-6 months
                        """
                        )

                    with col2:
                        st.metric("Market Share", f"{cluster.percentage:.1f}%")
                        st.metric("Revenue Potential", "High")
        else:
            st.info("No high priority segments")

        st.markdown("---")

        # 🟡 MEDIUM PRIORITY CLUSTERS
        st.markdown("#### 🟡 MEDIUM PRIORITY ACTIONS")
        medium_priority = [i for i in insights if "Medium" in i.priority]

        if medium_priority:
            for cluster in medium_priority:
                with st.expander(
                    f"Cluster {cluster.cluster_id} - {cluster.size:,} customers ({cluster.percentage:.1f}%)",
                    expanded=False,
                ):
                    col1, col2 = st.columns([2, 1])

                    with col1:
                        st.markdown(
                            f"""
                        **Segment Profile:**
                        - Age: {cluster.avg_age:.1f} years ({cluster.age_group})
                        - Income: BDT {cluster.avg_income:,.0f} ({cluster.income_group})
                        - Spending: {cluster.avg_spending:.1f}/100 ({cluster.spending_group})

                        **Strategy:**
                        {cluster.recommendation}

                        **Expected Impact:** Moderate
                        **Timeline:** 6-12 months
                        """
                        )

                    with col2:
                        st.metric("Market Share", f"{cluster.percentage:.1f}%")
                        st.metric("Revenue Potential", "Moderate")
        else:
            st.info("No medium priority segments")

        st.markdown("---")

        # 🟢 LOW PRIORITY CLUSTERS
        st.markdown("#### 🟢 LOW PRIORITY ACTIONS")
        low_priority = [i for i in insights if "Low" in i.priority]

        if low_priority:
            for cluster in low_priority:
                with st.expander(
                    f"Cluster {cluster.cluster_id} - {cluster.size:,} customers ({cluster.percentage:.1f}%)",
                    expanded=False,
                ):
                    col1, col2 = st.columns([2, 1])

                    with col1:
                        st.markdown(
                            f"""
                        **Segment Profile:**
                        - Age: {cluster.avg_age:.1f} years ({cluster.age_group})
                        - Income: BDT {cluster.avg_income:,.0f} ({cluster.income_group})
                        - Spending: {cluster.avg_spending:.1f}/100 ({cluster.spending_group})

                        **Strategy:**
                        {cluster.recommendation}

                        **Expected Impact:** Low
                        **Timeline:** Ongoing maintenance
                        """
                        )

                    with col2:
                        st.metric("Market Share", f"{cluster.percentage:.1f}%")
                        st.metric("Revenue Potential", "Low")
        else:
            st.info("No low priority segments")

        st.markdown("---")

        # BUDGET ALLOCATION
        st.markdown("### 💰 Recommended Budget Allocation")

        budget_data = {
            "Priority": ["High Priority", "Medium Priority", "Low Priority"],
            "Budget %": [50, 35, 15],
            "Reason": [
                "Maximum ROI focus",
                "Optimization efforts",
                "Maintenance campaigns",
            ],
        }
        st.dataframe(budget_data, use_container_width=True)

        st.markdown("---")

        # EXPORT FUNCTIONALITY
        st.markdown("### 📥 Export Results")

        col1, col2 = st.columns(2)

        with col1:
            result = st.session_state.get("clustering_result")
            if result:
                export_df = self.df.copy()
                export_df["Cluster"] = result.labels
                csv = export_df.to_csv(index=False)
                st.download_button(
                    label="📊 Download Cluster Data (CSV)",
                    data=csv,
                    file_name="customer_segmentation_results.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

        with col2:
            insights_df = pd.DataFrame(
                [
                    {
                        "Cluster": i.cluster_id,
                        "Size": i.size,
                        "Market %": f"{i.percentage:.1f}%",
                        "Avg Age": f"{i.avg_age:.1f}",
                        "Avg Income": f"BDT {i.avg_income:,.0f}",
                        "Avg Spending": f"{i.avg_spending:.1f}",
                        "Priority": i.priority,
                        "Strategy": i.recommendation,
                    }
                    for i in insights
                ]
            )
            csv = insights_df.to_csv(index=False)
            st.download_button(
                label="💡 Download Insights (CSV)",
                data=csv,
                file_name="cluster_insights_summary.csv",
                mime="text/csv",
                use_container_width=True,
            )

    def _render_footer(self) -> None:
        """Render footer."""
        st.markdown("---")
        st.markdown(
            """
        <div style='text-align: center; color: #667; padding: 20px;'>
            <p><strong>Customer Segmentation Dashboard v2.0.0</strong></p>
            <p>Developed by Shadin</p>
            <p>💡Built with Streamlit • Production-Ready • ML-Powered Analytics</p>
        </div>
        """,
            unsafe_allow_html=True,
        )


# APPLICATION ENTRY POINT


def main() -> None:
    """Application entry point."""
    configure_page()
    dashboard = CustomerSegmentationDashboard()
    dashboard.render()


if __name__ == "__main__":
    main()
