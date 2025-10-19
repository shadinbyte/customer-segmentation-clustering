"""
Customer Segmentation Analysis using Multiple Clustering Algorithms
Improved version with realistic data handling and validation

Author: BSc CSE Final Year Project
Date: October 2025
"""

import os
import warnings
from pathlib import Path

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
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# Configuration
plt.style.use("default")
sns.set_palette("husl")
OUTPUT_DIR = "clustering_results_3000"
DATA_FILE = "Synthetic_Mall_Customers_3000.csv"
VALIDATION_FILE = "Synthetic_Mall_Customers_3000_with_labels.csv"


class CustomerSegmentation:
    """
    Comprehensive Customer Segmentation Analysis
    Supports multiple clustering algorithms with validation
    """

    def __init__(self, data_path, validation_path=None):
        """
        Initialize the Customer Segmentation analysis

        Args:
            data_path: Path to main dataset CSV
            validation_path: Optional path to dataset with true labels
        """
        self.data_path = data_path
        self.validation_path = validation_path
        self.df = None
        self.df_validation = None
        self.scaler = StandardScaler()
        self.results = {}
        self.features = ["Age", "Annual Income (BDT)", "Spending Score (1-100)"]

        # Load data
        self._load_data()

    def _load_data(self):
        """Load and validate dataset"""
        try:
            self.df = pd.read_csv(self.data_path)
            print(f"Dataset loaded successfully from '{self.data_path}'")

            # Validate required columns
            required_cols = [
                "CustomerID",
                "Gender",
                "Age",
                "Annual Income (BDT)",
                "Spending Score (1-100)",
            ]
            missing_cols = set(required_cols) - set(self.df.columns)
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")

            # Load validation data if available
            if self.validation_path and Path(self.validation_path).exists():
                self.df_validation = pd.read_csv(self.validation_path)
                print(f"Validation data loaded from '{self.validation_path}'")
            else:
                print("No validation data available")

        except FileNotFoundError:
            raise FileNotFoundError(
                f"Dataset not found at '{self.data_path}'. "
                "Please run generator.py first to create the dataset."
            )
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")

    def preprocess_data(self):
        """Preprocess the data for clustering"""
        print("\n" + "=" * 60)
        print("STEP 1: DATA PREPROCESSING")
        print("=" * 60)

        # Display basic info
        print(f"\nDataset Shape: {self.df.shape}")
        print(f"Features: {self.df.columns.tolist()}")

        print("\nFirst 5 rows:")
        print(self.df.head())

        print("\nDataset Info:")
        print(self.df.info())

        print("\nDescriptive Statistics:")
        print(self.df[self.features].describe())

        # Check for missing values
        missing = self.df.isnull().sum()
        if missing.sum() > 0:
            print(f"\nWARNING - Missing Values:\n{missing[missing > 0]}")
            print("Dropping rows with missing values...")
            self.df = self.df.dropna()
        else:
            print("\nNo missing values detected")

        # Prepare features for clustering
        self.X = self.df[self.features].copy()

        # Scale the features
        self.X_scaled = self.scaler.fit_transform(self.X)

        print(f"\nFeatures used for clustering: {self.features}")
        print(f"Scaled data shape: {self.X_scaled.shape}")
        print(f"Scaling method: StandardScaler (mean=0, std=1)")

        return self.X_scaled

    def exploratory_data_analysis(self):
        """Perform exploratory data analysis"""
        print("\n" + "=" * 60)
        print("STEP 2: EXPLORATORY DATA ANALYSIS")
        print("=" * 60)

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(
            "Exploratory Data Analysis - 3000 Customers", fontsize=16, fontweight="bold"
        )

        # Distribution of Age
        axes[0, 0].hist(
            self.df["Age"], bins=30, alpha=0.7, color="skyblue", edgecolor="black"
        )
        axes[0, 0].set_title("Age Distribution", fontweight="bold")
        axes[0, 0].set_xlabel("Age (years)")
        axes[0, 0].set_ylabel("Frequency")
        axes[0, 0].grid(True, alpha=0.3)

        # Distribution of Annual Income
        axes[0, 1].hist(
            self.df["Annual Income (BDT)"],
            bins=30,
            alpha=0.7,
            color="lightgreen",
            edgecolor="black",
        )
        axes[0, 1].set_title("Annual Income Distribution", fontweight="bold")
        axes[0, 1].set_xlabel("Annual Income (BDT)")
        axes[0, 1].set_ylabel("Frequency")
        axes[0, 1].grid(True, alpha=0.3)

        # Distribution of Spending Score
        axes[0, 2].hist(
            self.df["Spending Score (1-100)"],
            bins=30,
            alpha=0.7,
            color="lightcoral",
            edgecolor="black",
        )
        axes[0, 2].set_title("Spending Score Distribution", fontweight="bold")
        axes[0, 2].set_xlabel("Spending Score")
        axes[0, 2].set_ylabel("Frequency")
        axes[0, 2].grid(True, alpha=0.3)

        # Gender distribution
        gender_counts = self.df["Gender"].value_counts()
        colors_pie = ["#87CEEB", "#FFB6C1"]
        axes[1, 0].pie(
            gender_counts.values,
            labels=gender_counts.index,
            autopct="%1.1f%%",
            colors=colors_pie,
            startangle=90,
        )
        axes[1, 0].set_title("Gender Distribution", fontweight="bold")

        # Income vs Spending Score
        scatter = axes[1, 1].scatter(
            self.df["Annual Income (BDT)"],
            self.df["Spending Score (1-100)"],
            c=self.df["Age"],
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
            self.df["Age"],
            self.df["Spending Score (1-100)"],
            alpha=0.5,
            color="purple",
            s=30,
        )
        axes[1, 2].set_title("Age vs Spending Score", fontweight="bold")
        axes[1, 2].set_xlabel("Age (years)")
        axes[1, 2].set_ylabel("Spending Score")
        axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            f"{OUTPUT_DIR}/EDA_3000_customers.png", dpi=300, bbox_inches="tight"
        )
        plt.show()
        print(f"EDA visualization saved to '{OUTPUT_DIR}/EDA_3000_customers.png'")

        # Correlation heatmap
        plt.figure(figsize=(10, 8))
        correlation_matrix = self.df[self.features].corr()
        sns.heatmap(
            correlation_matrix,
            annot=True,
            fmt=".3f",
            cmap="coolwarm",
            center=0,
            square=True,
            linewidths=1,
            cbar_kws={"label": "Correlation Coefficient"},
        )
        plt.title(
            "Correlation Heatmap - Feature Relationships",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )
        plt.tight_layout()
        plt.savefig(
            f"{OUTPUT_DIR}/correlation_heatmap_3000.png", dpi=300, bbox_inches="tight"
        )
        plt.show()
        print(
            f"Correlation heatmap saved to '{OUTPUT_DIR}/correlation_heatmap_3000.png'"
        )

        print("\nKey Observations:")
        print(f"- Age range: {self.df['Age'].min()} to {self.df['Age'].max()} years")
        print(
            f"- Income range: BDT {self.df['Annual Income (BDT)'].min():,} to BDT {self.df['Annual Income (BDT)'].max():,}"
        )
        print(
            f"- Spending Score range: {self.df['Spending Score (1-100)'].min()} to {self.df['Spending Score (1-100)'].max()}"
        )
        print(f"- Gender distribution: {dict(gender_counts)}")

    def find_optimal_kmeans_clusters(self, k_range=range(2, 11)):
        """Find optimal number of clusters for K-Means using multiple methods"""
        print("\n" + "=" * 60)
        print("STEP 3: OPTIMAL CLUSTER DETERMINATION")
        print("=" * 60)

        wcss = []  # Within-Cluster Sum of Squares
        silhouette_scores = []
        davies_bouldin_scores = []
        calinski_harabasz_scores = []

        print("\nTesting K-Means with different cluster numbers...")
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(self.X_scaled)
            wcss.append(kmeans.inertia_)

            # Calculate evaluation metrics
            if len(set(labels)) > 1:
                silhouette_scores.append(silhouette_score(self.X_scaled, labels))
                davies_bouldin_scores.append(
                    davies_bouldin_score(self.X_scaled, labels)
                )
                calinski_harabasz_scores.append(
                    calinski_harabasz_score(self.X_scaled, labels)
                )
            else:
                silhouette_scores.append(-1)
                davies_bouldin_scores.append(float("inf"))
                calinski_harabasz_scores.append(0)

            print(
                f"  k={k}: Silhouette={silhouette_scores[-1]:.3f}, "
                f"Davies-Bouldin={davies_bouldin_scores[-1]:.3f}, "
                f"Calinski-Harabasz={calinski_harabasz_scores[-1]:.0f}"
            )

        # Plot all metrics
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(
            "Optimal Cluster Determination for K-Means", fontsize=14, fontweight="bold"
        )

        # Elbow curve
        axes[0, 0].plot(k_range, wcss, "bo-", linewidth=2, markersize=8)
        axes[0, 0].set_xlabel("Number of Clusters (k)")
        axes[0, 0].set_ylabel("WCSS (Within-Cluster Sum of Squares)")
        axes[0, 0].set_title("Elbow Method")
        axes[0, 0].grid(True, alpha=0.3)

        # Silhouette scores
        axes[0, 1].plot(k_range, silhouette_scores, "ro-", linewidth=2, markersize=8)
        axes[0, 1].set_xlabel("Number of Clusters (k)")
        axes[0, 1].set_ylabel("Silhouette Score")
        axes[0, 1].set_title("Silhouette Analysis (Higher is Better)")
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].axhline(
            y=max(silhouette_scores),
            color="g",
            linestyle="--",
            alpha=0.5,
            label=f"Max: {max(silhouette_scores):.3f}",
        )
        axes[0, 1].legend()

        # Davies-Bouldin Index
        axes[1, 0].plot(
            k_range, davies_bouldin_scores, "go-", linewidth=2, markersize=8
        )
        axes[1, 0].set_xlabel("Number of Clusters (k)")
        axes[1, 0].set_ylabel("Davies-Bouldin Index")
        axes[1, 0].set_title("Davies-Bouldin Analysis (Lower is Better)")
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].axhline(
            y=min(davies_bouldin_scores),
            color="r",
            linestyle="--",
            alpha=0.5,
            label=f"Min: {min(davies_bouldin_scores):.3f}",
        )
        axes[1, 0].legend()

        # Calinski-Harabasz Index
        axes[1, 1].plot(
            k_range, calinski_harabasz_scores, "mo-", linewidth=2, markersize=8
        )
        axes[1, 1].set_xlabel("Number of Clusters (k)")
        axes[1, 1].set_ylabel("Calinski-Harabasz Index")
        axes[1, 1].set_title("Calinski-Harabasz Analysis (Higher is Better)")
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].axhline(
            y=max(calinski_harabasz_scores),
            color="b",
            linestyle="--",
            alpha=0.5,
            label=f"Max: {max(calinski_harabasz_scores):.0f}",
        )
        axes[1, 1].legend()

        plt.tight_layout()
        plt.savefig(
            f"{OUTPUT_DIR}/optimal_k_analysis.png", dpi=300, bbox_inches="tight"
        )
        plt.show()
        print(f"\nOptimal k analysis saved to '{OUTPUT_DIR}/optimal_k_analysis.png'")

        # Determine optimal k
        optimal_k_silhouette = list(k_range)[np.argmax(silhouette_scores)]
        optimal_k_db = list(k_range)[np.argmin(davies_bouldin_scores)]
        optimal_k_ch = list(k_range)[np.argmax(calinski_harabasz_scores)]

        print("\n" + "-" * 60)
        print("OPTIMAL K RECOMMENDATIONS:")
        print(f"  Based on Silhouette Score: k = {optimal_k_silhouette}")
        print(f"  Based on Davies-Bouldin Index: k = {optimal_k_db}")
        print(f"  Based on Calinski-Harabasz Index: k = {optimal_k_ch}")

        # Use majority vote or silhouette as tiebreaker
        optimal_k = optimal_k_silhouette
        print(f"\n  SELECTED OPTIMAL K: {optimal_k}")
        print("-" * 60)

        return optimal_k

    def kmeans_clustering(self, n_clusters=5):
        """Perform K-Means clustering"""
        print("\n" + "=" * 60)
        print(f"STEP 4: K-MEANS CLUSTERING (k={n_clusters})")
        print("=" * 60)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, max_iter=300)
        kmeans_labels = kmeans.fit_predict(self.X_scaled)

        # Calculate metrics
        silhouette_avg = silhouette_score(self.X_scaled, kmeans_labels)
        db_index = davies_bouldin_score(self.X_scaled, kmeans_labels)
        calinski_score = calinski_harabasz_score(self.X_scaled, kmeans_labels)

        print(f"\nK-Means Results:")
        print(f"  Number of clusters: {n_clusters}")
        print(f"  Silhouette Score: {silhouette_avg:.4f}")
        print(f"  Davies-Bouldin Index: {db_index:.4f} (lower is better)")
        print(f"  Calinski-Harabasz Index: {calinski_score:.2f} (higher is better)")
        print(f"  Iterations to converge: {kmeans.n_iter_}")
        print(f"  Inertia (WCSS): {kmeans.inertia_:.2f}")

        # Validation against true labels if available
        if self.df_validation is not None:
            ari = self._validate_against_true_labels(kmeans_labels, "K-Means")
            validation_score = ari
        else:
            validation_score = None

        self.results["KMeans"] = {
            "labels": kmeans_labels,
            "silhouette": silhouette_avg,
            "davies_bouldin": db_index,
            "calinski": calinski_score,
            "model": kmeans,
            "validation_score": validation_score,
        }

        self._plot_clusters(kmeans_labels, "K-Means Clustering")
        self._cluster_profiling(kmeans_labels, "K-Means")

        return kmeans_labels

    def hierarchical_clustering(self, n_clusters=5):
        """Perform Hierarchical Clustering"""
        print("\n" + "=" * 60)
        print(f"STEP 5: HIERARCHICAL CLUSTERING (k={n_clusters})")
        print("=" * 60)

        hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
        hierarchical_labels = hierarchical.fit_predict(self.X_scaled)

        # Calculate metrics
        silhouette_avg = silhouette_score(self.X_scaled, hierarchical_labels)
        db_index = davies_bouldin_score(self.X_scaled, hierarchical_labels)
        calinski_score = calinski_harabasz_score(self.X_scaled, hierarchical_labels)

        print(f"\nHierarchical Clustering Results:")
        print(f"  Number of clusters: {n_clusters}")
        print(f"  Silhouette Score: {silhouette_avg:.4f}")
        print(f"  Davies-Bouldin Index: {db_index:.4f} (lower is better)")
        print(f"  Calinski-Harabasz Index: {calinski_score:.2f} (higher is better)")
        print(f"  Linkage method: Ward")

        # Validation against true labels if available
        if self.df_validation is not None:
            ari = self._validate_against_true_labels(
                hierarchical_labels, "Hierarchical"
            )
            validation_score = ari
        else:
            validation_score = None

        self.results["Hierarchical"] = {
            "labels": hierarchical_labels,
            "silhouette": silhouette_avg,
            "davies_bouldin": db_index,
            "calinski": calinski_score,
            "model": hierarchical,
            "validation_score": validation_score,
        }

        self._plot_clusters(hierarchical_labels, "Hierarchical Clustering")
        self._cluster_profiling(hierarchical_labels, "Hierarchical")

        # Plot dendrogram
        self._plot_dendrogram()

        return hierarchical_labels

    def dbscan_clustering(self, eps=None, min_samples=None):
        """Perform DBSCAN clustering with automatic parameter optimization"""
        print("\n" + "=" * 60)
        print("STEP 6: DBSCAN CLUSTERING")
        print("=" * 60)

        # Auto-optimize parameters if not provided
        if eps is None or min_samples is None:
            print("\nAuto-optimizing DBSCAN parameters...")
            eps, min_samples = self._optimize_dbscan_params()

        print(f"\nUsing parameters: eps={eps}, min_samples={min_samples}")

        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        dbscan_labels = dbscan.fit_predict(self.X_scaled)

        # Count clusters (excluding noise)
        n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
        n_noise = list(dbscan_labels).count(-1)

        print(f"\nDBSCAN Results:")
        print(f"  Number of clusters found: {n_clusters}")
        print(f"  Number of noise points: {n_noise}")
        print(f"  Noise percentage: {(n_noise/len(dbscan_labels))*100:.2f}%")

        if n_clusters > 1:
            # Filter out noise for metric calculation
            mask = dbscan_labels != -1
            silhouette_avg = silhouette_score(self.X_scaled[mask], dbscan_labels[mask])
            db_index = davies_bouldin_score(self.X_scaled[mask], dbscan_labels[mask])
            calinski_score = calinski_harabasz_score(
                self.X_scaled[mask], dbscan_labels[mask]
            )
        else:
            silhouette_avg = -1
            db_index = float("inf")
            calinski_score = 0

        print(f"  Silhouette Score: {silhouette_avg:.4f}")
        print(f"  Davies-Bouldin Index: {db_index:.4f}")
        print(f"  Calinski-Harabasz Index: {calinski_score:.2f}")

        # Validation against true labels if available
        if self.df_validation is not None and n_clusters > 1:
            ari = self._validate_against_true_labels(dbscan_labels, "DBSCAN")
            validation_score = ari
        else:
            validation_score = None

        self.results["DBSCAN"] = {
            "labels": dbscan_labels,
            "silhouette": silhouette_avg,
            "davies_bouldin": db_index,
            "calinski": calinski_score,
            "model": dbscan,
            "n_clusters": n_clusters,
            "n_noise": n_noise,
            "validation_score": validation_score,
        }

        self._plot_clusters(dbscan_labels, "DBSCAN Clustering")
        if n_clusters > 0:
            self._cluster_profiling(dbscan_labels, "DBSCAN")

        return dbscan_labels

    def _optimize_dbscan_params(self):
        """Optimize DBSCAN parameters using grid search"""
        from sklearn.neighbors import NearestNeighbors

        # Use k-distance graph to estimate eps
        k = 10
        neighbors = NearestNeighbors(n_neighbors=k)
        neighbors.fit(self.X_scaled)
        distances, _ = neighbors.kneighbors(self.X_scaled)
        distances = np.sort(distances[:, -1])

        # Use elbow point (roughly at 95th percentile)
        eps_optimal = np.percentile(distances, 95)

        # Min samples typically 2*dimensions for DBSCAN
        min_samples_optimal = 2 * self.X_scaled.shape[1]

        print(f"  Estimated eps: {eps_optimal:.3f}")
        print(f"  Estimated min_samples: {min_samples_optimal}")

        return eps_optimal, min_samples_optimal

    def gaussian_mixture_clustering(self, n_components=5):
        """Perform Gaussian Mixture Model clustering"""
        print("\n" + "=" * 60)
        print(f"STEP 7: GAUSSIAN MIXTURE MODEL (n_components={n_components})")
        print("=" * 60)

        gmm = GaussianMixture(n_components=n_components, random_state=42, max_iter=200)
        gmm_labels = gmm.fit_predict(self.X_scaled)

        # Calculate metrics
        silhouette_avg = silhouette_score(self.X_scaled, gmm_labels)
        db_index = davies_bouldin_score(self.X_scaled, gmm_labels)
        calinski_score = calinski_harabasz_score(self.X_scaled, gmm_labels)

        print(f"\nGaussian Mixture Model Results:")
        print(f"  Number of components: {n_components}")
        print(f"  Silhouette Score: {silhouette_avg:.4f}")
        print(f"  Davies-Bouldin Index: {db_index:.4f} (lower is better)")
        print(f"  Calinski-Harabasz Index: {calinski_score:.2f} (higher is better)")
        print(f"  Converged: {gmm.converged_}")
        print(f"  BIC (Bayesian Information Criterion): {gmm.bic(self.X_scaled):.2f}")
        print(f"  AIC (Akaike Information Criterion): {gmm.aic(self.X_scaled):.2f}")

        # Validation against true labels if available
        if self.df_validation is not None:
            ari = self._validate_against_true_labels(gmm_labels, "GMM")
            validation_score = ari
        else:
            validation_score = None

        self.results["GMM"] = {
            "labels": gmm_labels,
            "silhouette": silhouette_avg,
            "davies_bouldin": db_index,
            "calinski": calinski_score,
            "model": gmm,
            "probabilities": gmm.predict_proba(self.X_scaled),
            "validation_score": validation_score,
        }

        self._plot_clusters(gmm_labels, "Gaussian Mixture Model")
        self._cluster_profiling(gmm_labels, "GMM")

        return gmm_labels

    def _validate_against_true_labels(self, predicted_labels, algorithm_name):
        """Validate clustering results against true labels"""
        if self.df_validation is None:
            return None

        true_labels = self.df_validation["True_Segment"]

        # Convert string labels to numeric
        unique_labels = {label: idx for idx, label in enumerate(true_labels.unique())}
        true_labels_numeric = true_labels.map(unique_labels)

        # Calculate Adjusted Rand Index
        ari = adjusted_rand_score(true_labels_numeric, predicted_labels)

        print(f"\n  VALIDATION - Adjusted Rand Index: {ari:.4f}")
        print(f"  (1.0 = perfect match, 0.0 = random, <0 = worse than random)")

        return ari

    def _plot_clusters(self, labels, algorithm_name):
        """Plot clustering results"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(
            f"{algorithm_name} - 3000 Customers", fontsize=14, fontweight="bold"
        )

        # Determine unique clusters and colors
        unique_labels = sorted(set(labels))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

        # Handle noise points in DBSCAN
        if -1 in unique_labels:
            colors = np.vstack([[0.5, 0.5, 0.5, 0.3], colors[1:]])

        # Plot 1: Income vs Spending Score
        for idx, label in enumerate(unique_labels):
            mask = labels == label
            label_name = f"Cluster {label}" if label != -1 else "Noise"
            axes[0].scatter(
                self.df.loc[mask, "Annual Income (BDT)"],
                self.df.loc[mask, "Spending Score (1-100)"],
                c=[colors[idx]],
                label=label_name,
                alpha=0.6,
                s=50,
                edgecolors="black",
                linewidths=0.5,
            )
        axes[0].set_xlabel("Annual Income (BDT)")
        axes[0].set_ylabel("Spending Score (1-100)")
        axes[0].set_title(f"{algorithm_name} - Income vs Spending")
        axes[0].legend(loc="best")
        axes[0].grid(True, alpha=0.3)

        # Plot 2: Age vs Spending Score
        for idx, label in enumerate(unique_labels):
            mask = labels == label
            label_name = f"Cluster {label}" if label != -1 else "Noise"
            axes[1].scatter(
                self.df.loc[mask, "Age"],
                self.df.loc[mask, "Spending Score (1-100)"],
                c=[colors[idx]],
                label=label_name,
                alpha=0.6,
                s=50,
                edgecolors="black",
                linewidths=0.5,
            )
        axes[1].set_xlabel("Age (years)")
        axes[1].set_ylabel("Spending Score (1-100)")
        axes[1].set_title(f"{algorithm_name} - Age vs Spending")
        axes[1].legend(loc="best")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        filename = f'{algorithm_name.replace(" ", "_").lower()}_clusters.png'
        plt.savefig(f"{OUTPUT_DIR}/{filename}", dpi=300, bbox_inches="tight")
        plt.show()
        print(f"\nCluster visualization saved to '{OUTPUT_DIR}/{filename}'")

    def _plot_dendrogram(self, sample_size=100):
        """Plot dendrogram for hierarchical clustering"""
        print(f"\nGenerating dendrogram (using {sample_size} samples for clarity)...")

        plt.figure(figsize=(16, 8))

        # Use a smaller subset for clearer visualization
        sample_indices = np.random.choice(
            len(self.X_scaled), sample_size, replace=False
        )
        Z = linkage(self.X_scaled[sample_indices], "ward")

        plt.title(
            f"Hierarchical Clustering Dendrogram (Sample: {sample_size} customers)",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )
        plt.xlabel("Sample Index", fontsize=12)
        plt.ylabel("Distance (Ward Linkage)", fontsize=12)

        # Remove x-axis labels to avoid clutter - show only tree structure
        dendrogram(Z, no_labels=True, color_threshold=10)

        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/dendrogram.png", dpi=300, bbox_inches="tight")
        plt.show()
        print(f"Dendrogram saved to '{OUTPUT_DIR}/dendrogram.png'")

    def _cluster_profiling(self, labels, algorithm_name):
        """Create detailed cluster profiles with business interpretations"""
        print(f"\n" + "-" * 60)
        print(f"{algorithm_name} - CLUSTER PROFILING")
        print("-" * 60)

        # Add labels to dataframe for analysis
        temp_df = self.df.copy()
        temp_df["Cluster"] = labels

        # Calculate cluster statistics
        cluster_stats = []

        for cluster_id in sorted(set(labels)):
            if cluster_id == -1:  # Skip noise points in DBSCAN
                cluster_data = temp_df[temp_df["Cluster"] == cluster_id]
                print(
                    f"\nCluster {cluster_id} (Noise): {len(cluster_data)} customers ({len(cluster_data)/len(temp_df)*100:.1f}%)"
                )
                continue

            cluster_data = temp_df[temp_df["Cluster"] == cluster_id]
            cluster_size = len(cluster_data)

            # Calculate statistics
            age_mean = cluster_data["Age"].mean()
            age_std = cluster_data["Age"].std()
            income_mean = cluster_data["Annual Income (BDT)"].mean()
            income_std = cluster_data["Annual Income (BDT)"].std()
            spending_mean = cluster_data["Spending Score (1-100)"].mean()
            spending_std = cluster_data["Spending Score (1-100)"].std()

            # Gender distribution
            gender_mode = cluster_data["Gender"].mode()
            dominant_gender = gender_mode[0] if len(gender_mode) > 0 else "Mixed"
            female_pct = (cluster_data["Gender"] == "Female").sum() / cluster_size * 100

            # Business interpretation
            age_group = (
                "Young"
                if age_mean < 30
                else "Middle-aged" if age_mean < 50 else "Senior"
            )
            income_group = (
                "Low"
                if income_mean < 500000
                else "Medium" if income_mean < 1000000 else "High"
            )
            spending_group = (
                "Low"
                if spending_mean < 40
                else "Medium" if spending_mean < 70 else "High"
            )

            cluster_stats.append(
                {
                    "Cluster": cluster_id,
                    "Size": cluster_size,
                    "Percentage": f"{cluster_size/len(temp_df)*100:.1f}%",
                    "Avg_Age": f"{age_mean:.1f}",
                    "Avg_Income": f"{income_mean:,.0f}",
                    "Avg_Spending": f"{spending_mean:.1f}",
                    "Dominant_Gender": dominant_gender,
                    "Female_%": f"{female_pct:.1f}%",
                }
            )

            # Print detailed profile
            print(
                f"\nCluster {cluster_id}: {cluster_size} customers ({cluster_size/len(temp_df)*100:.1f}%)"
            )
            print(f"  Demographics:")
            print(f"    - Age: {age_mean:.1f} ± {age_std:.1f} years ({age_group})")
            print(f"    - Gender: {female_pct:.1f}% Female, {100-female_pct:.1f}% Male")
            print(f"  Financial Profile:")
            print(
                f"    - Income: BDT {income_mean:,.0f} ± {income_std:,.0f} ({income_group})"
            )
            print(
                f"    - Spending Score: {spending_mean:.1f} ± {spending_std:.1f} ({spending_group})"
            )
            print(
                f"  Business Segment: {age_group}, {income_group} income, {spending_group} spending"
            )

            # Marketing recommendations
            recommendation = self._generate_marketing_strategy(
                age_group, income_group, spending_group, cluster_size, len(temp_df)
            )
            print(f"  Strategy: {recommendation}")

        # Create summary table
        if cluster_stats:
            summary_df = pd.DataFrame(cluster_stats)
            print(f"\n{algorithm_name} - Summary Table:")
            print(summary_df.to_string(index=False))

        return cluster_stats

    def _generate_marketing_strategy(
        self, age_group, income_group, spending_group, cluster_size, total_customers
    ):
        """Generate targeted marketing strategy based on cluster characteristics"""

        market_share = cluster_size / total_customers * 100

        if spending_group == "High" and income_group == "High":
            return (
                "Premium products, VIP programs, exclusive offers, personalized service"
            )
        elif spending_group == "High" and income_group in ["Medium", "Low"]:
            return (
                "Value bundles, loyalty rewards, installment plans, quality assurance"
            )
        elif spending_group == "Low" and income_group == "High":
            return "Trust-building campaigns, product demonstrations, value proposition focus"
        elif spending_group == "Medium":
            if market_share > 20:
                return "Mass market campaigns, seasonal promotions, volume discounts"
            else:
                return "Targeted engagement, personalized offers, cross-selling opportunities"
        else:
            return "Entry-level products, first-purchase discounts, education campaigns"

    def compare_algorithms(self):
        """Compare performance of all clustering algorithms"""
        print("\n" + "=" * 60)
        print("STEP 8: ALGORITHM COMPARISON")
        print("=" * 60)

        comparison_data = []
        for algo_name, result in self.results.items():
            # Skip DBSCAN if it didn't find meaningful clusters
            if algo_name == "DBSCAN" and result.get("n_clusters", 0) <= 1:
                print(
                    f"\nSkipping {algo_name} from comparison (insufficient clusters found)"
                )
                continue

            comparison_data.append(
                {
                    "Algorithm": algo_name,
                    "Silhouette": f"{result['silhouette']:.4f}",
                    "Davies_Bouldin": f"{result['davies_bouldin']:.4f}",
                    "Calinski_Harabasz": f"{result['calinski']:.0f}",
                    "Clusters": len(set(result["labels"]))
                    - (1 if -1 in result["labels"] else 0),
                    "Validation_ARI": (
                        f"{result.get('validation_score', 0):.4f}"
                        if result.get("validation_score") is not None
                        else "N/A"
                    ),
                }
            )

        comparison_df = pd.DataFrame(comparison_data)

        print("\nAlgorithm Performance Comparison:")
        print(comparison_df.to_string(index=False))

        # Determine best algorithm
        comparison_df["Silhouette_num"] = comparison_df["Silhouette"].astype(float)
        best_algo = comparison_df.loc[
            comparison_df["Silhouette_num"].idxmax(), "Algorithm"
        ]
        print(f"\nBest performing algorithm (by Silhouette Score): {best_algo}")

        # Plot comparison
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle("Algorithm Performance Comparison", fontsize=14, fontweight="bold")

        algorithms = comparison_df["Algorithm"]

        # Silhouette Scores
        silhouette_scores = comparison_df["Silhouette"].astype(float)
        bars1 = axes[0].bar(
            algorithms,
            silhouette_scores,
            color="lightblue",
            alpha=0.7,
            edgecolor="navy",
            linewidth=2,
        )
        axes[0].set_title("Silhouette Score\n(Higher is Better)", fontweight="bold")
        axes[0].set_ylabel("Silhouette Score")
        axes[0].tick_params(axis="x", rotation=45)
        axes[0].grid(True, alpha=0.3, axis="y")

        for bar, score in zip(bars1, silhouette_scores):
            axes[0].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{score:.3f}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        # Davies-Bouldin Index
        db_scores = comparison_df["Davies_Bouldin"].astype(float)
        bars2 = axes[1].bar(
            algorithms,
            db_scores,
            color="lightcoral",
            alpha=0.7,
            edgecolor="darkred",
            linewidth=2,
        )
        axes[1].set_title("Davies-Bouldin Index\n(Lower is Better)", fontweight="bold")
        axes[1].set_ylabel("Davies-Bouldin Index")
        axes[1].tick_params(axis="x", rotation=45)
        axes[1].grid(True, alpha=0.3, axis="y")

        for bar, score in zip(bars2, db_scores):
            axes[1].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{score:.3f}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        # Calinski-Harabasz Index
        ch_scores = comparison_df["Calinski_Harabasz"].astype(float)
        bars3 = axes[2].bar(
            algorithms,
            ch_scores,
            color="lightgreen",
            alpha=0.7,
            edgecolor="darkgreen",
            linewidth=2,
        )
        axes[2].set_title(
            "Calinski-Harabasz Index\n(Higher is Better)", fontweight="bold"
        )
        axes[2].set_ylabel("Calinski-Harabasz Index")
        axes[2].tick_params(axis="x", rotation=45)
        axes[2].grid(True, alpha=0.3, axis="y")

        for bar, score in zip(bars3, ch_scores):
            axes[2].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 50,
                f"{score:.0f}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        plt.tight_layout()
        plt.savefig(
            f"{OUTPUT_DIR}/algorithm_comparison.png", dpi=300, bbox_inches="tight"
        )
        plt.show()
        print(f"\nComparison chart saved to '{OUTPUT_DIR}/algorithm_comparison.png'")

        return comparison_df

    def save_results(self):
        """Save all clustering results to CSV files"""
        print("\n" + "=" * 60)
        print("STEP 9: SAVING RESULTS")
        print("=" * 60)

        # Create results directory if it doesn't exist
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # Save dataset with all cluster labels
        results_df = self.df.copy()

        for algo_name, result in self.results.items():
            results_df[f"{algo_name}_Cluster"] = result["labels"]

        # Save to CSV
        output_file = f"{OUTPUT_DIR}/all_clustering_results.csv"
        results_df.to_csv(output_file, index=False)
        print(f"\nAll clustering results saved to '{output_file}'")

        # Save comparison table
        comparison_df = self.compare_algorithms()
        comparison_file = f"{OUTPUT_DIR}/algorithm_comparison.csv"
        comparison_df.to_csv(comparison_file, index=False)
        print(f"Algorithm comparison saved to '{comparison_file}'")

        # Save detailed metrics
        metrics_data = []
        for algo_name, result in self.results.items():
            metrics_data.append(
                {
                    "Algorithm": algo_name,
                    "Silhouette_Score": result["silhouette"],
                    "Davies_Bouldin_Index": result["davies_bouldin"],
                    "Calinski_Harabasz_Index": result["calinski"],
                    "Number_of_Clusters": len(set(result["labels"]))
                    - (1 if -1 in result["labels"] else 0),
                    "Validation_ARI": result.get("validation_score", "N/A"),
                }
            )

        metrics_df = pd.DataFrame(metrics_data)
        metrics_file = f"{OUTPUT_DIR}/detailed_metrics.csv"
        metrics_df.to_csv(metrics_file, index=False)
        print(f"Detailed metrics saved to '{metrics_file}'")

        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETED SUCCESSFULLY!")
        print("=" * 60)

    def generate_thesis_results_section(self):
        """Generate formatted results for thesis inclusion"""
        print("\n" + "=" * 60)
        print("THESIS RESULTS SECTION (Copy to your thesis)")
        print("=" * 60)

        print("\n### CLUSTERING RESULTS TABLE\n")
        print(
            "| Algorithm        | Silhouette | Davies-Bouldin | Calinski-Harabasz | Clusters | Validation ARI |"
        )
        print(
            "|------------------|------------|----------------|-------------------|----------|----------------|"
        )

        for algo_name, result in self.results.items():
            n_clusters = len(set(result["labels"])) - (
                1 if -1 in result["labels"] else 0
            )
            ari = result.get("validation_score", None)
            ari_str = f"{ari:.3f}" if ari is not None else "N/A"

            print(
                f"| {algo_name:16} | {result['silhouette']:10.3f} | "
                f"{result['davies_bouldin']:14.3f} | {result['calinski']:17.0f} | "
                f"{n_clusters:8} | {ari_str:14} |"
            )

        print("\n### KEY FINDINGS\n")

        # Find best algorithm
        best_algo = max(
            self.results.items(),
            key=lambda x: x[1]["silhouette"] if x[1].get("n_clusters", 0) > 1 else -1,
        )

        print(f"1. Best performing algorithm: {best_algo[0]}")
        print(f"   - Silhouette Score: {best_algo[1]['silhouette']:.4f}")
        print(f"   - Davies-Bouldin Index: {best_algo[1]['davies_bouldin']:.4f}")
        print(f"   - Calinski-Harabasz Index: {best_algo[1]['calinski']:.2f}")

        if best_algo[1].get("validation_score") is not None:
            print(f"   - Validation ARI: {best_algo[1]['validation_score']:.4f}")

        print(f"\n2. Total customers analyzed: {len(self.df)}")
        print(f"3. Features used: {', '.join(self.features)}")
        print(
            f"4. Number of clusters identified: {len(set(best_algo[1]['labels'])) - (1 if -1 in best_algo[1]['labels'] else 0)}"
        )

        print("\n" + "=" * 60)


def main():
    """Main execution function"""
    print("=" * 70)
    print(" " * 10 + "CUSTOMER SEGMENTATION ANALYSIS - 3000 CUSTOMERS")
    print("=" * 70)
    print("\nBSc CSE Final Year Project")
    print("Clustering Algorithms: K-Means, Hierarchical, DBSCAN, Gaussian Mixture")
    print("=" * 70)

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Initialize the analysis
    try:
        analysis = CustomerSegmentation(
            data_path=DATA_FILE, validation_path=VALIDATION_FILE
        )
    except Exception as e:
        print(f"\nERROR: {e}")
        print("\nPlease ensure:")
        print("1. You have run generator.py to create the dataset")
        print("2. The CSV files are in the correct location")
        return

    # Step 1: Preprocess data
    X_scaled = analysis.preprocess_data()

    # Step 2: Exploratory Data Analysis
    analysis.exploratory_data_analysis()

    # Step 3: Find optimal K for K-Means
    optimal_k = analysis.find_optimal_kmeans_clusters()

    # Step 4-7: Apply all clustering algorithms
    print("\n" + "=" * 70)
    print("RUNNING ALL CLUSTERING ALGORITHMS")
    print("=" * 70)

    kmeans_labels = analysis.kmeans_clustering(n_clusters=optimal_k)
    hierarchical_labels = analysis.hierarchical_clustering(n_clusters=optimal_k)
    dbscan_labels = analysis.dbscan_clustering()  # Auto-optimize parameters
    gmm_labels = analysis.gaussian_mixture_clustering(n_components=optimal_k)

    # Step 8: Compare algorithms
    comparison_results = analysis.compare_algorithms()

    # Step 9: Save results
    analysis.save_results()

    # Generate thesis section
    analysis.generate_thesis_results_section()

    print("\n" + "=" * 70)
    print("ALL ANALYSIS COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print(f"\nGenerated Files in '{OUTPUT_DIR}/' directory:")
    print("  - EDA_3000_customers.png")
    print("  - correlation_heatmap_3000.png")
    print("  - optimal_k_analysis.png")
    print("  - k-means_clustering_clusters.png")
    print("  - hierarchical_clustering_clusters.png")
    print("  - dbscan_clustering_clusters.png")
    print("  - gaussian_mixture_model_clusters.png")
    print("  - dendrogram.png")
    print("  - algorithm_comparison.png")
    print("  - all_clustering_results.csv")
    print("  - algorithm_comparison.csv")
    print("  - detailed_metrics.csv")
    print("\nNext Steps:")
    print("  1. Review all visualizations in the output directory")
    print("  2. Copy the thesis results section to your thesis document")
    print("  3. Run the dashboard: streamlit run customer_segmentation_dashboard.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
