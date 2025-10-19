# 🛍️ Customer Segmentation using Clustering Algorithms

Ever wondered how businesses know exactly what their customers want? This project tackles that problem using machine learning.

I built a complete customer segmentation system that analyzes shopping behavior and automatically groups customers into meaningful segments. Think of it as giving businesses "customer X-ray vision" - they can see patterns that aren't obvious at first glance.

**Built as my final year BSc project in Computer Science & Engineering**

---

## What Does This Actually Do?

Imagine you're running a shopping mall with 3,000 customers. Some are young high-spenders who love premium brands. Others are budget-conscious families looking for deals. Without data, you'd treat everyone the same way - wasting money on marketing that doesn't work.

This system solves that problem by:

- Automatically finding distinct customer groups based on their behavior
- Telling you exactly who these customers are (age, income, spending habits)
- Suggesting specific marketing strategies for each group

No guesswork. Just data-driven insights.

---

## The Results (TL;DR)

- **Analyzed**: 3,000 mall customers
- **Tested**: 4 different machine learning algorithms
- **Found**: 5 distinct customer segments
- **Accuracy**: 96.89% validation score
- **Best Algorithm**: K-Means (though I tested GMM, Hierarchical, and DBSCAN too)

The system identified groups like "Young Affluent Spenders" and "Senior Budget-Conscious Shoppers" - each needing completely different marketing approaches.

---

## Live Demo

Here's what the interactive dashboard looks like in action:

![Customer Segmentation Dashboard](screenshots/dashboard_main.png)

_The dashboard lets you switch between algorithms, see clusters in 3D, and get instant business insights - no coding required._

---

## Why I Built This

During my studies, I noticed a gap: most academic projects use tiny datasets (500-800 customers) and stop at technical results. I wanted to build something that:

1. **Actually scales** - 3,000 customers is realistic for mid-sized businesses
2. **Compares properly** - testing 4 algorithms to see what really works best
3. **Provides real value** - not just clustering scores, but actual marketing strategies
4. **Anyone can use** - built a dashboard so non-technical people can explore the data

Basically, I wanted to bridge the gap between "cool ML project" and "something a business could actually deploy tomorrow."

---

## The Customer Segments We Discovered

After running the analysis, here's what emerged:

### 1. Young Affluent Spenders (20% of customers)

- **Profile**: Average age 25, income BDT 1.2M, very high spending (85/100)
- **What they want**: Premium products, exclusive experiences
- **Strategy**: VIP programs, early access to new products, luxury branding

### 2. Middle-Income Families (22%)

- **Profile**: Age 41, income BDT 513K, moderate spending (47/100)
- **What they want**: Value for money, family-friendly options
- **Strategy**: Bundle deals, seasonal sales, family packages

### 3. High-Income Conservatives (20%)

- **Profile**: Age 49, income BDT 1.4M, surprisingly low spending (35/100)
- **What they want**: Trust, quality assurance, value demonstration
- **Strategy**: This is the "untapped potential" group - focus on building trust

### 4. Senior Budget-Conscious (18%)

- **Profile**: Age 57, income BDT 891K, minimal spending (22/100)
- **What they want**: Affordability, simplicity, discounts
- **Strategy**: Senior discounts, straightforward messaging, value products

### 5. Value-Seeking Shoppers (21%)

- **Profile**: Age 31, income BDT 749K, high engagement (72/100)
- **What they want**: Quality at reasonable prices
- **Strategy**: Loyalty programs, quality emphasis, exclusive member benefits

---

## How It Works (The Technical Bit)

### The Algorithms I Compared

I didn't just use one algorithm and call it done. I tested four different approaches:

**K-Means** (Winner - 0.549 Silhouette Score)

- Fastest, most reliable
- Works great when customer groups are relatively distinct
- Industry standard for a reason

**Gaussian Mixture Models** (Close second - 0.549)

- Almost identical performance to K-Means
- Gives probability scores ("this customer is 80% likely Segment A")
- Useful when customers fit multiple segments

**Hierarchical Clustering** (Solid - 0.546)

- Creates a family tree of customer relationships
- Great for understanding how segments relate to each other
- Slightly slower but very interpretable

**DBSCAN** (Struggled - didn't work well here)

- Found everything as one big cluster
- Taught me an important lesson: algorithm choice matters!
- Great for other types of data, just not this one

### Validation: How I Know It Actually Works

I didn't just trust the algorithms blindly. I validated results three ways:

1. **Silhouette Score** (0.549) - measures how well-separated clusters are
2. **Davies-Bouldin Index** (0.651) - lower is better, checks cluster quality
3. **Calinski-Harabasz** (6,504) - higher is better, measures separation vs cohesion
4. **Ground Truth Comparison** (96.89% ARI) - checked against known customer types

All metrics agreed: K-Means found real, meaningful patterns.

---

## Quick Start

Want to run this yourself? Here's how:

### Installation

```bash
# Clone the repo
git clone https://github.com/shadinbyte/customer-segmentation-clustering.git
cd customer-segmentation-clustering

# Install requirements
pip install -r requirements.txt
```

### Generate Sample Data

```bash
python generator.py
```

This creates realistic synthetic customer data. I used synthetic data for privacy reasons - no real customer information here.

### Run the Analysis

```bash
python customer_segmentation_analysis.py
```

Sit back for 2-3 minutes. The script will:

- Run all 4 algorithms
- Generate 12 visualization charts
- Calculate all quality metrics
- Create segment profiles with business recommendations
- Save everything to `clustering_results_3000/`

### Launch the Dashboard

```bash
streamlit run customer_segmentation_dashboard.py
```

Opens in your browser at `http://localhost:8501`

Now you can:

- Switch between algorithms live
- Adjust parameters and see instant results
- Rotate 3D visualizations
- Export results as CSV
- Show it to your non-technical boss!

---

## Project Structure

```
customer-segmentation-clustering/
├── customer_segmentation_analysis.py   # Main analysis engine
├── customer_segmentation_dashboard.py  # Interactive dashboard
├── generator.py                        # Creates realistic test data
├── requirements.txt                    # Python dependencies
├── clustering_results_3000/            # All outputs go here
│   ├── *.png                          # Visualizations
│   └── *.csv                          # Results & metrics
└── README.md                           # You are here!
```

Everything is self-contained. No complex setup, no database required.

---

## What I Learned Building This

**Technical Skills:**

- How different ML algorithms actually behave with real-ish data
- Why feature scaling matters so much (spoiler: distance calculations)
- The importance of using multiple validation metrics
- Building dashboards that non-coders can actually use

**Practical Insights:**

- DBSCAN failed because customer data doesn't have "density valleys"
- K-Means++ initialization is way better than random (faster convergence)
- Silhouette scores alone don't tell the full story
- Business interpretability matters as much as technical accuracy

**What I'd Do Differently:**

- Add temporal analysis (how do customers move between segments over time?)
- Integrate with a real CRM system instead of CSV files
- Build a REST API so marketing tools could query segments automatically
- Add A/B testing framework to measure if segmentation actually improves campaigns

---

## Technologies Used

**Core:**

- Python 3.8+
- Scikit-learn (the ML heavy lifting)
- Pandas & NumPy (data wrangling)

**Visualization:**

- Matplotlib & Seaborn (static charts)
- Plotly (interactive 3D plots)

**Dashboard:**

- Streamlit (turns Python into a web app)

**Development:**

- Git/GitHub (version control)
- Jupyter Notebooks (experimentation)

---

## Files You'll Get After Running

The analysis generates everything you need:

**Visualizations** (all 300 DPI, publication-ready)

- Exploratory data analysis charts
- Correlation heatmaps
- Optimal K determination plots
- Cluster visualizations (one per algorithm)
- Algorithm comparison charts
- Dendrogram (hierarchical relationships)

**Data Files**

- Complete results with cluster assignments
- Algorithm performance comparison
- Detailed metrics for each algorithm

**Business Intelligence**

- Segment profiles with demographics
- Marketing strategy recommendations
- Customer counts and distributions

---

## Results at a Glance

| Algorithm        | Silhouette Score | Davies-Bouldin | Calinski-Harabasz | Segments Found |
| ---------------- | ---------------- | -------------- | ----------------- | -------------- |
| **K-Means**      | **0.549**        | 0.651          | 6,504             | 5              |
| Gaussian Mixture | 0.549            | 0.651          | 6,496             | 5              |
| Hierarchical     | 0.546            | 0.655          | 6,438             | 5              |
| DBSCAN           | -1.000           | ∞              | 0                 | 1\*            |

\*DBSCAN found only 1 dense cluster, demonstrating algorithm selection matters

**Key Finding**: K-Means and Gaussian Mixture Models performed nearly identically, while DBSCAN struggled with this data structure - highlighting the importance of choosing the right algorithm for your data.

---

## Real-World Applications

This isn't just an academic exercise. Here's how businesses could use this:

**Retail:**

- Personalized email campaigns per segment
- Store layout optimization (premium section vs budget section)
- Inventory planning (stock what each segment wants)

**E-commerce:**

- Product recommendation engines
- Dynamic pricing strategies
- Targeted ads on social media

**Services:**

- Tiered service offerings
- Customized loyalty programs
- Resource allocation (focus sales team on high-value segments)

**General:**

- Customer lifetime value prediction
- Churn risk identification
- New market entry strategies

---

## Why This Matters for Hiring

If you're a hiring manager looking at this, here's what this project demonstrates:

✅ **I can handle real-scale data** - 3,000 records isn't trivial
✅ **I don't just accept defaults** - compared 4 algorithms systematically
✅ **I validate properly** - used multiple metrics, not just one
✅ **I think about end users** - built a dashboard, not just scripts
✅ **I understand business context** - translated clusters into strategies
✅ **I document clearly** - you're reading this, aren't you?
✅ **I know when things fail** - DBSCAN didn't work, I explained why

I'm not just a coder. I solve problems end-to-end.

---

## Future Enhancements (The Roadmap)

If I had more time, here's what I'd add:

**Short-term:**

- [ ] CSV upload feature (analyze your own data)
- [ ] PDF report generation
- [ ] Email alerts for segment changes

**Medium-term:**

- [ ] REST API for programmatic access
- [ ] Integration with Google Analytics
- [ ] Automated A/B testing framework

**Long-term:**

- [ ] Real-time clustering as new customers arrive
- [ ] Deep learning approaches (autoencoders)
- [ ] Predictive segment migration (who's likely to move to premium?)

---

## Questions?

**Q: Can I use my own data?**
A: Yes! Just format it as CSV with Age, Income, and Spending Score columns. The code handles the rest.

**Q: Why only 3 features?**
A: Simplicity and visualization. The framework easily extends to more features.

**Q: Why synthetic data?**
A: Privacy and reproducibility. Anyone can run this. Real customer data would require NDAs.

**Q: Which algorithm should I actually use?**
A: For customer segmentation like this? K-Means. It's fast, reliable, and interpretable.

**Q: Can this scale to 100,000 customers?**
A: K-Means yes. Hierarchical no (too slow). DBSCAN maybe. GMM probably.

---

## License

MIT License - use it however you want. Build something cool and tell me about it!

---

## Acknowledgments

Thanks to my thesis supervisor [Supervisor Name] for guidance throughout this project.

Inspired by real-world customer analytics challenges in Bangladesh's growing retail sector.

---

**⭐ If this helped you understand customer segmentation or you learned something, drop a star!**
