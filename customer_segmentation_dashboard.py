"""
Customer Segmentation Interactive Dashboard
Streamlit-based visualization and exploration tool

Author: BSc CSE Final Year Project
Date: October 2025
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
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

# Page configuration
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
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
</style>
""",
    unsafe_allow_html=True,
)


class CustomerSegmentationDashboard:
    def __init__(self):
        self.df = None
        self.df_validation = None
        self.X_scaled = None
        self.scaler = StandardScaler()
        self.results = {}
        self.features = ["Age", "Annual Income (BDT)", "Spending Score (1-100)"]

    def load_data(self):
        """Load and preprocess the dataset"""
        try:
            self.df = pd.read_csv("Synthetic_Mall_Customers_3000.csv")

            # Try to load validation data
            try:
                self.df_validation = pd.read_csv(
                    "Synthetic_Mall_Customers_3000_with_labels.csv"
                )
                st.sidebar.success("Validation data loaded")
            except:
                st.sidebar.info("Validation data not available")

            # Display dataset info in sidebar
            st.sidebar.markdown("### 📊 Dataset Info")
            st.sidebar.write(f"**Total Customers:** {len(self.df):,}")
            st.sidebar.write(f"**Features:** {len(self.df.columns)}")

            gender_counts = self.df["Gender"].value_counts()
            st.sidebar.write(f"**Male:** {gender_counts.get('Male', 0):,}")
            st.sidebar.write(f"**Female:** {gender_counts.get('Female', 0):,}")

            # Prepare features for clustering
            self.X = self.df[self.features]

            # Scale the features
            self.X_scaled = self.scaler.fit_transform(self.X)

            return True
        except Exception as e:
            st.error(f"Error loading data: {e}")
            st.info(
                "Please ensure 'Synthetic_Mall_Customers_3000.csv' exists in the directory"
            )
            return False

    def perform_clustering(self, algorithm, n_clusters=5, eps=0.5, min_samples=5):
        """Perform clustering based on selected algorithm"""
        try:
            if algorithm == "K-Means":
                model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                labels = model.fit_predict(self.X_scaled)

            elif algorithm == "Hierarchical":
                model = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
                labels = model.fit_predict(self.X_scaled)

            elif algorithm == "DBSCAN":
                model = DBSCAN(eps=eps, min_samples=min_samples)
                labels = model.fit_predict(self.X_scaled)

            elif algorithm == "Gaussian Mixture":
                model = GaussianMixture(n_components=n_clusters, random_state=42)
                labels = model.fit_predict(self.X_scaled)

            # Calculate metrics
            n_clusters_found = len(set(labels)) - (1 if -1 in labels else 0)

            if n_clusters_found > 1:
                # Filter out noise for metrics if DBSCAN
                if -1 in labels:
                    mask = labels != -1
                    silhouette_avg = silhouette_score(self.X_scaled[mask], labels[mask])
                    db_index = davies_bouldin_score(self.X_scaled[mask], labels[mask])
                    calinski_score = calinski_harabasz_score(
                        self.X_scaled[mask], labels[mask]
                    )
                else:
                    silhouette_avg = silhouette_score(self.X_scaled, labels)
                    db_index = davies_bouldin_score(self.X_scaled, labels)
                    calinski_score = calinski_harabasz_score(self.X_scaled, labels)
            else:
                silhouette_avg = -1
                db_index = float("inf")
                calinski_score = 0

            # Calculate validation ARI if available
            ari = None
            if self.df_validation is not None and n_clusters_found > 1:
                true_labels = self.df_validation["True_Segment"]
                unique_labels = {
                    label: idx for idx, label in enumerate(true_labels.unique())
                }
                true_labels_numeric = true_labels.map(unique_labels)
                ari = adjusted_rand_score(true_labels_numeric, labels)

            return (
                labels,
                model,
                silhouette_avg,
                db_index,
                calinski_score,
                n_clusters_found,
                ari,
            )

        except Exception as e:
            st.error(f"Clustering error: {e}")
            return None, None, -1, float("inf"), 0, 0, None

    def create_cluster_visualizations(self, labels, algorithm_name):
        """Create interactive cluster visualizations"""
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Income vs Spending Score",
                "Age vs Spending Score",
                "Age vs Income",
                "Cluster Distribution",
            ),
            horizontal_spacing=0.1,
            vertical_spacing=0.12,
        )

        unique_clusters = sorted(set(labels))
        colors = px.colors.qualitative.Set1

        for i, cluster in enumerate(unique_clusters):
            if cluster == -1:
                cluster_name = "Noise"
                color = "gray"
            else:
                cluster_name = f"Cluster {cluster}"
                color = colors[i % len(colors)]

            cluster_mask = labels == cluster
            cluster_size = np.sum(cluster_mask)

            # Income vs Spending
            fig.add_trace(
                go.Scatter(
                    x=self.df[cluster_mask]["Annual Income (BDT)"],
                    y=self.df[cluster_mask]["Spending Score (1-100)"],
                    mode="markers",
                    name=f"{cluster_name} (n={cluster_size})",
                    marker=dict(size=6, opacity=0.7, color=color),
                    legendgroup=f"cluster_{cluster}",
                    showlegend=True,
                ),
                row=1,
                col=1,
            )

            # Age vs Spending
            fig.add_trace(
                go.Scatter(
                    x=self.df[cluster_mask]["Age"],
                    y=self.df[cluster_mask]["Spending Score (1-100)"],
                    mode="markers",
                    name=cluster_name,
                    marker=dict(size=6, opacity=0.7, color=color),
                    legendgroup=f"cluster_{cluster}",
                    showlegend=False,
                ),
                row=1,
                col=2,
            )

            # Age vs Income
            fig.add_trace(
                go.Scatter(
                    x=self.df[cluster_mask]["Age"],
                    y=self.df[cluster_mask]["Annual Income (BDT)"],
                    mode="markers",
                    name=cluster_name,
                    marker=dict(size=6, opacity=0.7, color=color),
                    legendgroup=f"cluster_{cluster}",
                    showlegend=False,
                ),
                row=2,
                col=1,
            )

        # Cluster distribution
        cluster_counts = pd.Series(labels).value_counts().sort_index()
        cluster_names = [
            f"Cluster {c}" if c != -1 else "Noise" for c in cluster_counts.index
        ]

        fig.add_trace(
            go.Bar(
                x=cluster_names,
                y=cluster_counts.values,
                name="Cluster Size",
                marker_color=[
                    colors[i % len(colors)] if cluster_counts.index[i] != -1 else "gray"
                    for i in range(len(cluster_counts))
                ],
                showlegend=False,
            ),
            row=2,
            col=2,
        )

        fig.update_layout(
            height=700,
            title_text=f"{algorithm_name} Clustering - 3000 Customers",
            title_x=0.5,
            showlegend=True,
        )

        # Update axis labels
        fig.update_xaxes(title_text="Annual Income (BDT)", row=1, col=1)
        fig.update_yaxes(title_text="Spending Score", row=1, col=1)
        fig.update_xaxes(title_text="Age", row=1, col=2)
        fig.update_yaxes(title_text="Spending Score", row=1, col=2)
        fig.update_xaxes(title_text="Age", row=2, col=1)
        fig.update_yaxes(title_text="Annual Income (BDT)", row=2, col=1)
        fig.update_xaxes(title_text="Cluster", row=2, col=2)
        fig.update_yaxes(title_text="Number of Customers", row=2, col=2)

        return fig

    def generate_cluster_insights(self, labels):
        """Generate business insights for each cluster"""
        temp_df = self.df.copy()
        temp_df["Cluster"] = labels

        insights = []

        for cluster_id in sorted(set(labels)):
            if cluster_id == -1:
                continue

            cluster_data = temp_df[temp_df["Cluster"] == cluster_id]
            cluster_size = len(cluster_data)

            # Calculate statistics
            avg_age = cluster_data["Age"].mean()
            avg_income = cluster_data["Annual Income (BDT)"].mean()
            avg_spending = cluster_data["Spending Score (1-100)"].mean()
            dominant_gender = (
                cluster_data["Gender"].mode()[0]
                if len(cluster_data["Gender"].mode()) > 0
                else "Unknown"
            )

            # Business interpretation
            age_group = (
                "Young" if avg_age < 30 else "Middle-aged" if avg_age < 50 else "Senior"
            )
            income_group = (
                "Low"
                if avg_income < 500000
                else "Medium" if avg_income < 1000000 else "High"
            )
            spending_group = (
                "Low"
                if avg_spending < 40
                else "Medium" if avg_spending < 70 else "High"
            )

            # Generate recommendations
            if spending_group == "High" and income_group == "High":
                recommendation = "Premium products, VIP programs, exclusive offers"
                priority = "High"
            elif spending_group == "High" and income_group in ["Medium", "Low"]:
                recommendation = "Value bundles, loyalty rewards, quality products"
                priority = "Medium"
            elif spending_group == "Low" and income_group == "High":
                recommendation = "Trust-building, product demonstrations, value focus"
                priority = "Medium"
            else:
                recommendation = "Standard campaigns, seasonal promotions"
                priority = "Low"

            insights.append(
                {
                    "cluster": cluster_id,
                    "size": cluster_size,
                    "percentage": (cluster_size / len(temp_df)) * 100,
                    "avg_age": avg_age,
                    "avg_income": avg_income,
                    "avg_spending": avg_spending,
                    "dominant_gender": dominant_gender,
                    "age_group": age_group,
                    "income_group": income_group,
                    "spending_group": spending_group,
                    "recommendation": recommendation,
                    "priority": priority,
                }
            )

        return insights


def main():
    st.markdown(
        '<h1 style="text-align: center; color: #1f77b4;">🛍️ Customer Segmentation Dashboard</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<h3 style="text-align: center;">Advanced Analysis of 3000 Customers</h3>',
        unsafe_allow_html=True,
    )

    # Initialize dashboard
    dashboard = CustomerSegmentationDashboard()

    # Load data
    if not dashboard.load_data():
        st.stop()

    # Sidebar configuration
    st.sidebar.markdown("## ⚙️ Algorithm Settings")

    algorithm = st.sidebar.selectbox(
        "Select Clustering Algorithm",
        ["K-Means", "Hierarchical", "DBSCAN", "Gaussian Mixture"],
        help="Choose the clustering algorithm to analyze customer data",
    )

    if algorithm in ["K-Means", "Hierarchical", "Gaussian Mixture"]:
        n_clusters = st.sidebar.slider("Number of Clusters", 2, 10, 5)
        eps, min_samples = 0.5, 5
    else:
        eps = st.sidebar.slider("EPS (Neighborhood radius)", 0.1, 1.0, 0.36, 0.01)
        min_samples = st.sidebar.slider("Min Samples", 3, 20, 6)
        n_clusters = 5

    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 Data Overview", "🔍 Clustering", "💡 Insights", "📈 Action Plan"]
    )

    with tab1:
        st.header("Dataset Overview - 3000 Customers")

        # Key metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Customers", f"{len(dashboard.df):,}")
        with col2:
            st.metric("Average Age", f"{dashboard.df['Age'].mean():.1f} years")
        with col3:
            st.metric(
                "Average Income",
                f"BDT {dashboard.df['Annual Income (BDT)'].mean():,.0f}",
            )
        with col4:
            st.metric(
                "Avg Spending Score",
                f"{dashboard.df['Spending Score (1-100)'].mean():.1f}",
            )

        # Data exploration
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("Data Sample")
            st.dataframe(dashboard.df.head(10), use_container_width=True)

            st.subheader("Feature Distributions")
            selected_feature = st.selectbox(
                "Select feature to visualize", dashboard.features
            )
            fig_hist = px.histogram(
                dashboard.df,
                x=selected_feature,
                title=f"Distribution of {selected_feature}",
                color_discrete_sequence=["#1f77b4"],
                nbins=30,
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        with col2:
            st.subheader("Quick Statistics")
            st.dataframe(
                dashboard.df[dashboard.features].describe(), use_container_width=True
            )

            st.subheader("Gender Distribution")
            gender_fig = px.pie(
                dashboard.df,
                names="Gender",
                title="Customer Gender Distribution",
                color_discrete_sequence=["#87CEEB", "#FFB6C1"],
            )
            st.plotly_chart(gender_fig, use_container_width=True)

    with tab2:
        st.header("Clustering Analysis")

        if st.button(
            "🎯 Run Clustering Analysis", type="primary", use_container_width=True
        ):
            with st.spinner(f"Running {algorithm} on 3000 customers..."):
                result = dashboard.perform_clustering(
                    algorithm, n_clusters, eps, min_samples
                )
                labels, model, silhouette, db_index, calinski, n_clusters_found, ari = (
                    result
                )

            if labels is not None:
                st.success(f"✅ {algorithm} clustering completed!")

                # Metrics
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("Clusters Found", n_clusters_found)
                with col2:
                    st.metric("Silhouette Score", f"{silhouette:.3f}")
                with col3:
                    st.metric("Davies-Bouldin", f"{db_index:.3f}")
                with col4:
                    st.metric("Calinski-Harabasz", f"{calinski:.0f}")
                with col5:
                    if ari is not None:
                        st.metric("Validation ARI", f"{ari:.3f}")
                    else:
                        st.metric("Validation ARI", "N/A")

                # Visualizations
                st.subheader("Cluster Visualization")
                fig_clusters = dashboard.create_cluster_visualizations(
                    labels, algorithm
                )
                st.plotly_chart(fig_clusters, use_container_width=True)

                # 3D Visualization
                st.subheader("3D Cluster View")
                fig_3d = px.scatter_3d(
                    dashboard.df,
                    x="Annual Income (BDT)",
                    y="Spending Score (1-100)",
                    z="Age",
                    color=labels.astype(str),
                    title=f"{algorithm} - 3D Visualization",
                    opacity=0.7,
                    color_discrete_sequence=px.colors.qualitative.Set1,
                    labels={"color": "Cluster"},
                )
                st.plotly_chart(fig_3d, use_container_width=True)

                # Store results in session state
                st.session_state["labels"] = labels
                st.session_state["algorithm"] = algorithm
        else:
            st.info("👆 Click the button above to run clustering analysis")

    with tab3:
        st.header("Business Insights")

        if "labels" in st.session_state:
            labels = st.session_state["labels"]
            algorithm = st.session_state["algorithm"]
            insights = dashboard.generate_cluster_insights(labels)

            st.subheader(f"Cluster Analysis - {algorithm}")

            for insight in insights:
                with st.expander(
                    f"Cluster {insight['cluster']} - {insight['size']} customers ({insight['percentage']:.1f}%)",
                    expanded=True,
                ):
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("**📈 Cluster Profile**")
                        st.write(
                            f"**Average Age:** {insight['avg_age']:.1f} years ({insight['age_group']})"
                        )
                        st.write(
                            f"**Average Income:** BDT {insight['avg_income']:,.0f} ({insight['income_group']})"
                        )
                        st.write(
                            f"**Spending Score:** {insight['avg_spending']:.1f} ({insight['spending_group']})"
                        )
                        st.write(f"**Dominant Gender:** {insight['dominant_gender']}")

                    with col2:
                        st.markdown("**🎯 Business Strategy**")
                        st.write(
                            f"**Segment:** {insight['age_group']}, {insight['income_group']} income"
                        )
                        st.success(f"**Recommendation:** {insight['recommendation']}")
                        st.write(f"**Priority:** {insight['priority']}")
        else:
            st.info("Run clustering analysis first to see insights")

    with tab4:
        st.header("Marketing Action Plan")

        if "labels" in st.session_state:
            st.subheader("📋 Recommended Strategies")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### 🎯 Targeted Campaigns")
                strategies = [
                    "**Personalized Email Marketing**: Segment-specific recommendations",
                    "**Loyalty Programs**: Reward high-spending customers",
                    "**Dynamic Pricing**: Adjust offers based on segments",
                    "**Cross-selling**: Recommend complementary products",
                    "**Retention Programs**: Focus on high-value customers",
                ]
                for strategy in strategies:
                    st.markdown(f"- {strategy}")

                st.markdown("### 💰 Revenue Optimization")
                revenue_tips = [
                    "**High-Value Focus**: Prioritize highest LTV clusters",
                    "**Upsell Opportunities**: Target medium-spending segments",
                    "**Budget Allocation**: Distribute spend by segment potential",
                    "**Seasonal Campaigns**: Time promotions by segment patterns",
                ]
                for tip in revenue_tips:
                    st.markdown(f"- {tip}")

            with col2:
                st.markdown("### 📊 Performance Metrics")
                metrics = [
                    "**Conversion Rates**: Track by segment",
                    "**Customer Lifetime Value**: Monitor LTV changes",
                    "**Retention Rates**: Measure by cluster",
                    "**Average Order Value**: Track spending patterns",
                    "**Customer Acquisition Cost**: Optimize CAC",
                ]
                for metric in metrics:
                    st.markdown(f"- {metric}")

                st.markdown("### 🔄 Continuous Improvement")
                improvement = [
                    "**Regular Re-clustering**: Update every 6 months",
                    "**A/B Testing**: Test strategies across clusters",
                    "**Feedback Loops**: Incorporate customer feedback",
                    "**AI Integration**: Real-time personalization",
                ]
                for item in improvement:
                    st.markdown(f"- {item}")

            # Export section
            st.markdown("---")
            st.subheader("📥 Export Results")

            export_df = dashboard.df.copy()
            export_df["Cluster"] = st.session_state["labels"]

            csv = export_df.to_csv(index=False)
            st.download_button(
                label="Download Cluster Data (CSV)",
                data=csv,
                file_name="customer_segmentation_results.csv",
                mime="text/csv",
                use_container_width=True,
            )
        else:
            st.info("Run clustering analysis first to generate action plans")


if __name__ == "__main__":
    main()
