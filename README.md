# SmartCart — Customer Segmentation

Unsupervised machine learning project that segments retail customers into distinct groups based on demographics, income, and spending behavior, to support targeted marketing and personalization strategies.

## Overview

This project analyzes a customer dataset (2,240 customers, 22 raw features) and groups customers into **4 behavioral segments** using clustering techniques. The resulting segments can be used to tailor marketing campaigns, promotions, and product recommendations to each customer group.

## Project Pipeline

1. **Data Cleaning**
   - Handle missing values (median imputation for `Income`)
2. **Feature Engineering**
   - `Age` from `Year_Birth`
   - `Customer_Tenure_Days` from `Dt_Customer`
   - `Total_Spending` (sum across product categories)
   - `Total_Children` (`Kidhome` + `Teenhome`)
   - Simplified `Education` and `Living_With` categories
3. **Outlier Removal**
   - Filtered unrealistic ages (> 90) and income (> 600,000)
4. **Exploratory Data Analysis**
   - Pairplots and correlation heatmap
5. **Preprocessing**
   - One-hot encoding for categorical features
   - Standard scaling for numerical features
6. **Dimensionality Reduction**
   - PCA (3 components) for visualization and clustering input
7. **Optimal Cluster Selection**
   - Elbow method (WCSS + `KneeLocator`)
   - Silhouette score analysis
   - Optimal K = 4
8. **Clustering**
   - K-Means
   - Agglomerative Clustering (Ward linkage)
9. **Cluster Characterization**
   - Cluster distribution, income vs. spending patterns, per-cluster summary statistics

## Repository Structure

```
smartcart/
├── smartcart.ipynb        # Main analysis notebook
├── README.md               # Project documentation
├── requirements.txt         # Python dependencies
```

## Data

This project uses a customer marketing dataset with demographic and purchase-history fields (income, recency, spending across product categories, campaign responses, etc.). The raw data file is **not included in this repository**. If you're using a public dataset (e.g., Kaggle's "Customer Personality Analysis"), download it from the original source and place it at `data/smartcart_customers.csv`, respecting that source's license terms.

## Setup

```bash
git clone https://github.com/Anshuman14022005/SmartCart-Customer-Segmentation-System
cd smartcart
pip install -r requirements.txt
jupyter notebook smartcart.ipynb
```

## Requirements

- Python 3.9+
- pandas
- matplotlib
- seaborn
- scikit-learn
- kneed

## Results

Four customer segments were identified, differing primarily in income level and total spending, with clear separation visible in PCA space. See the notebook's final section (`Characterization of Clusters`) for per-cluster summary statistics.

