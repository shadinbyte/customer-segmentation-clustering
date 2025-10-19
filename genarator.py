"""
Realistic Customer Data Generator for Customer Segmentation Analysis
Generates 3000 customers with realistic correlations and patterns
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set seed for reproducibility
np.random.seed(42)


class RealisticCustomerGenerator:
    """Generate realistic customer data with meaningful patterns"""

    def __init__(self, n_customers=3000):
        self.n_customers = n_customers
        self.df = None

    def generate_customer_segments(self):
        """
        Generate 5 distinct customer segments with CLEAR SEPARATION
        Optimized for clustering algorithms to discover 5 distinct groups
        Reduced correlations to emphasize segment differences
        """

        # Define 5 customer segments with CLEAR SEPARATION
        segments = {
            "Young Premium Spenders": {
                "size_ratio": 0.20,  # 20% of customers
                "age_mean": 26,
                "age_std": 3,  # Tighter, younger
                "income_mean": 1250000,
                "income_std": 120000,  # High income, tight
                "spending_mean": 85,
                "spending_std": 7,  # Very high spending, tight
                "gender_ratio": 0.58,  # Slightly more female
            },
            "Senior Conservative": {
                "size_ratio": 0.18,
                "age_mean": 58,
                "age_std": 4,  # Much older, tight
                "income_mean": 850000,
                "income_std": 100000,  # Medium-high income
                "spending_mean": 22,
                "spending_std": 6,  # Very low spending, tight
                "gender_ratio": 0.42,  # Slightly more male
            },
            "Middle-aged Budget Shoppers": {
                "size_ratio": 0.22,
                "age_mean": 42,
                "age_std": 5,  # Middle age
                "income_mean": 500000,
                "income_std": 90000,  # Low income, tight
                "spending_mean": 48,
                "spending_std": 8,  # Medium spending
                "gender_ratio": 0.52,
            },
            "High-income Cautious": {
                "size_ratio": 0.20,
                "age_mean": 50,
                "age_std": 5,  # Upper middle age
                "income_mean": 1350000,
                "income_std": 110000,  # Very high income
                "spending_mean": 35,
                "spending_std": 7,  # Low-medium spending
                "gender_ratio": 0.45,
            },
            "Young Active Spenders": {
                "size_ratio": 0.20,
                "age_mean": 32,
                "age_std": 4,  # Young-middle
                "income_mean": 750000,
                "income_std": 100000,  # Medium income
                "spending_mean": 72,
                "spending_std": 8,  # High spending
                "gender_ratio": 0.55,
            },
        }

        all_data = []
        customer_id = 1

        for segment_name, params in segments.items():
            n_segment = int(self.n_customers * params["size_ratio"])

            # Generate age with tight bounds (less correlation, more separation)
            age = np.random.normal(params["age_mean"], params["age_std"], n_segment)
            age = np.clip(age, 18, 70).astype(int)

            # Generate income with MINIMAL correlation to age (to emphasize segments)
            # Reduced age effect to 10% of previous strength
            age_factor = 1 + 0.002 * (age - 25) - 0.00008 * ((age - 45) ** 2)
            income = np.random.normal(
                params["income_mean"], params["income_std"], n_segment
            )
            income = income * age_factor
            income = np.clip(income, 150000, 1500000).astype(int)

            # Generate spending score with MINIMAL correlation (segment-driven)
            # Removed youth_factor and income_normalized to emphasize segment differences
            spending = np.random.normal(
                params["spending_mean"], params["spending_std"], n_segment
            )

            # Add only tiny random variation (2% effect instead of previous strong correlations)
            age_noise = (age - params["age_mean"]) / params["age_std"] * 0.5
            income_noise = (income - params["income_mean"]) / params["income_std"] * 0.3

            spending = spending + age_noise + income_noise
            spending = np.clip(spending, 1, 100).astype(int)

            # Generate gender with specified ratio
            n_female = int(n_segment * params["gender_ratio"])
            gender = ["Female"] * n_female + ["Male"] * (n_segment - n_female)
            np.random.shuffle(gender)

            # Create segment dataframe
            segment_df = pd.DataFrame(
                {
                    "CustomerID": range(customer_id, customer_id + n_segment),
                    "Gender": gender,
                    "Age": age,
                    "Annual Income (BDT)": income,
                    "Spending Score (1-100)": spending,
                    "True_Segment": segment_name,  # For validation only
                }
            )

            all_data.append(segment_df)
            customer_id += n_segment

        # Combine all segments
        self.df = pd.concat(all_data, ignore_index=True)

        # Shuffle to mix segments
        self.df = self.df.sample(frac=1, random_state=42).reset_index(drop=True)

        # Reassign CustomerID sequentially
        self.df["CustomerID"] = range(1, len(self.df) + 1)

        return self.df

    def add_realistic_noise(self):
        """Add small random noise to make data more realistic"""

        # Add small random variations
        self.df["Age"] += np.random.randint(-1, 2, len(self.df))
        self.df["Age"] = np.clip(self.df["Age"], 18, 70)

        self.df["Annual Income (BDT)"] += np.random.randint(-20000, 20000, len(self.df))
        self.df["Annual Income (BDT)"] = np.clip(
            self.df["Annual Income (BDT)"], 150000, 1500000
        )

        self.df["Spending Score (1-100)"] += np.random.randint(-3, 4, len(self.df))
        self.df["Spending Score (1-100)"] = np.clip(
            self.df["Spending Score (1-100)"], 1, 100
        )

        return self.df

    def visualize_generated_data(self):
        """Create visualizations to verify realistic patterns"""

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(
            "Generated Customer Data - Quality Check", fontsize=16, fontweight="bold"
        )

        # Age distribution
        axes[0, 0].hist(
            self.df["Age"], bins=30, alpha=0.7, color="skyblue", edgecolor="black"
        )
        axes[0, 0].set_title("Age Distribution")
        axes[0, 0].set_xlabel("Age")
        axes[0, 0].set_ylabel("Frequency")

        # Income distribution
        axes[0, 1].hist(
            self.df["Annual Income (BDT)"],
            bins=30,
            alpha=0.7,
            color="lightgreen",
            edgecolor="black",
        )
        axes[0, 1].set_title("Income Distribution")
        axes[0, 1].set_xlabel("Annual Income (BDT)")
        axes[0, 1].set_ylabel("Frequency")

        # Spending Score distribution
        axes[0, 2].hist(
            self.df["Spending Score (1-100)"],
            bins=30,
            alpha=0.7,
            color="lightcoral",
            edgecolor="black",
        )
        axes[0, 2].set_title("Spending Score Distribution")
        axes[0, 2].set_xlabel("Spending Score")
        axes[0, 2].set_ylabel("Frequency")

        # Income vs Spending (should show clusters)
        scatter = axes[1, 0].scatter(
            self.df["Annual Income (BDT)"],
            self.df["Spending Score (1-100)"],
            c=self.df["Age"],
            alpha=0.6,
            cmap="viridis",
            s=20,
        )
        axes[1, 0].set_title("Income vs Spending Score")
        axes[1, 0].set_xlabel("Annual Income (BDT)")
        axes[1, 0].set_ylabel("Spending Score")
        plt.colorbar(scatter, ax=axes[1, 0], label="Age")

        # Age vs Income (should show correlation)
        axes[1, 1].scatter(
            self.df["Age"],
            self.df["Annual Income (BDT)"],
            alpha=0.4,
            color="purple",
            s=20,
        )
        axes[1, 1].set_title("Age vs Income (Should show correlation)")
        axes[1, 1].set_xlabel("Age")
        axes[1, 1].set_ylabel("Annual Income (BDT)")

        # Age vs Spending
        axes[1, 2].scatter(
            self.df["Age"],
            self.df["Spending Score (1-100)"],
            alpha=0.4,
            color="orange",
            s=20,
        )
        axes[1, 2].set_title("Age vs Spending Score")
        axes[1, 2].set_xlabel("Age")
        axes[1, 2].set_ylabel("Spending Score")

        plt.tight_layout()
        plt.savefig("data_generation_validation.png", dpi=300, bbox_inches="tight")
        plt.show()

        print("\nValidation plot saved as 'data_generation_validation.png'")

    def print_statistics(self):
        """Print comprehensive statistics of generated data"""

        print("\n" + "=" * 60)
        print("REALISTIC CUSTOMER DATA GENERATION - STATISTICS")
        print("=" * 60)

        print(f"\nTotal Customers Generated: {len(self.df)}")
        print(f"Features: {len(self.df.columns) - 1}")  # Exclude True_Segment

        print("\n--- Gender Distribution ---")
        gender_counts = self.df["Gender"].value_counts()
        for gender, count in gender_counts.items():
            print(f"{gender}: {count} ({count/len(self.df)*100:.1f}%)")

        print("\n--- Feature Statistics ---")
        print(
            self.df[["Age", "Annual Income (BDT)", "Spending Score (1-100)"]].describe()
        )

        print("\n--- Correlation Matrix ---")
        correlation = self.df[
            ["Age", "Annual Income (BDT)", "Spending Score (1-100)"]
        ].corr()
        print(correlation)

        print("\n--- True Segment Distribution (for validation) ---")
        segment_counts = self.df["True_Segment"].value_counts()
        for segment, count in segment_counts.items():
            print(f"{segment}: {count} ({count/len(self.df)*100:.1f}%)")

        print("\n" + "=" * 60)

    def save_dataset(
        self, filename="Synthetic_Mall_Customers_3000.csv", include_true_labels=False
    ):
        """Save the generated dataset to CSV"""

        # Remove True_Segment column unless specified
        if include_true_labels:
            save_df = self.df.copy()
            # Save with labels for validation
            save_df.to_csv(filename.replace(".csv", "_with_labels.csv"), index=False)
            print(
                f"\nDataset with true labels saved to '{filename.replace('.csv', '_with_labels.csv')}'"
            )

        # Save without true labels (for clustering)
        save_df = self.df.drop("True_Segment", axis=1)
        save_df.to_csv(filename, index=False)

        print(f"Dataset saved to '{filename}'")
        print(f"Shape: {save_df.shape}")
        print(f"Columns: {list(save_df.columns)}")


def main():
    """Main execution function"""

    print("=" * 60)
    print("REALISTIC CUSTOMER DATA GENERATOR")
    print("=" * 60)

    # Initialize generator
    generator = RealisticCustomerGenerator(n_customers=3000)

    # Generate realistic customer segments
    print("\nStep 1: Generating customer segments with realistic patterns...")
    df = generator.generate_customer_segments()
    print(f"Generated {len(df)} customers across 5 segments")

    # Add realistic noise
    print("\nStep 2: Adding realistic variations...")
    df = generator.add_realistic_noise()

    # Print statistics
    generator.print_statistics()

    # Visualize data
    print("\nStep 3: Creating validation visualizations...")
    generator.visualize_generated_data()

    # Save datasets
    print("\nStep 4: Saving datasets...")
    generator.save_dataset(include_true_labels=True)

    print("\n" + "=" * 60)
    print("DATA GENERATION COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\nGenerated Files:")
    print("  - Synthetic_Mall_Customers_3000.csv (for clustering)")
    print("  - Synthetic_Mall_Customers_3000_with_labels.csv (for validation)")
    print("  - data_generation_validation.png (quality check)")
    print("\nNext Steps:")
    print("  1. Review 'data_generation_validation.png' to verify data quality")
    print("  2. Run customer_segmentation_analysis.py for clustering")
    print("  3. Compare results with true labels for validation")
    print("=" * 60)


if __name__ == "__main__":
    main()
