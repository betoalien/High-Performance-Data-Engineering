---
title: Chapter 5 - Interactive Data Engineering
parent: High-Performance Data Engineering
nav_order: 6
has_children: false
description: Jupyter notebooks for exploratory data analysis with the hybrid Rust+Python engine
---

# Chapter 5: Interactive Data Engineering (JupyterLab)

This chapter demonstrates how to use Jupyter notebooks for interactive data exploration using our hybrid Rust+Python DataFrame engine. We'll walk through a complete end-to-end workflow.

## Notebook Setup

### Installing Dependencies

```bash
# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or: .venv\Scripts\activate  # Windows

# Install the SDK
pip install -e hyperframe_sdk/

# Install Jupyter and visualization libraries
pip install jupyterlab matplotlib seaborn
```

### Starting Jupyter

```bash
jupyter lab --notebook-dir=./notebooks
```

## Notebook 1: Data Exploration Workflow

### Loading and Initial Inspection

```python
# Cell 1: Setup
from hyperframe import DataFrame, read_csv
import time

print("SDK loaded - Rust engine ready!")
```

```python
# Cell 2: Load dataset
df = read_csv("sales_data.csv")

print(f"Dataset shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"\nColumn names: {df.columns}")
```

```python
# Cell 3: Preview data
df.head(10)
```

```python
# Cell 4: Data types and null counts
df.info()
```

### Exploratory Analysis

```python
# Cell 5: Summary statistics for numeric columns
df.describe()
```

```python
# Cell 6: Unique value counts
print("Product categories:")
category_counts = df.group_by("category").agg({"price": "count"})
print(category_counts)

print("\nRegional distribution:")
region_counts = df.group_by("region").agg({"price": "count"})
print(region_counts)
```

```python
# Cell 7: Correlation analysis
# Export specific numeric columns to pandas for statistical functions
# (use Rust for scale, pandas for statistics on aggregated data)
numeric_pdf = df[["price", "quantity", "discount"]].to_pandas()
print(numeric_pdf.corr().round(3))
```

## Notebook 2: Data Transformation Pipeline

### Step-by-Step Transformation

```python
# Cell 1: Load raw data
raw_df = read_csv("raw_customer_data.csv")
print(f"Raw data: {raw_df.shape}")
```

```python
# Cell 2: Handle missing values
print("Null counts before cleaning:")
for col in raw_df.columns:
    null_count = raw_df[col].isnull().sum()
    if null_count > 0:
        print(f"  {col}: {null_count:,}")

# Fill strategies
raw_df.fill_na("quantity", 0)
raw_df.fill_na("discount", 0.0)
raw_df.fill_na("region", "Unknown")
clean_df = raw_df
```

```python
# Cell 3: Type conversions
clean_df["order_date"] = clean_df["order_date"].astype("Date")
clean_df["customer_id"] = clean_df["customer_id"].astype("Utf8")
```

```python
# Cell 4: Feature engineering
clean_df["revenue"] = clean_df["quantity"] * clean_df["price"] * (1 - clean_df["discount"])
clean_df["order_month"] = clean_df["order_date"].str.slice(0, 7)  # Extract "YYYY-MM"
clean_df["is_high_value"] = clean_df["revenue"] > 1000
```

```python
# Cell 5: Verify transformations
print(f"Clean data: {clean_df.shape}")
print(f"\nNew columns added: revenue, order_month, is_high_value")
clean_df.head()
```

## Notebook 3: Advanced Analytics

### Time Series Analysis

```python
# Cell 1: Monthly revenue trend
monthly = clean_df.group_by("order_month").agg({
    "revenue": "sum",
    "order_id": "count"
}).sort_by("order_month")

monthly.head(12)
```

```python
# Cell 2: Year-over-year growth
# Use pandas for statistical operations on the aggregated (small) result
monthly_pd = monthly.to_pandas()
monthly_pd["revenue_pct_change"] = monthly_pd["revenue"].pct_change(12) * 100
print(monthly_pd.tail(6))
```

### Customer Segmentation

```python
# Cell 3: RFM Analysis (Recency, Frequency, Monetary)

# Recency: Days since last order
recency = clean_df.group_by("customer_id").agg({
    "order_date": "max"
})

# Frequency: Number of orders
frequency = clean_df.group_by("customer_id").agg({
    "order_id": "count"
})

# Monetary: Total spend
monetary = clean_df.group_by("customer_id").agg({
    "revenue": "sum"
})

# Combine into RFM table
rfm = recency.join(frequency, on="customer_id") \
             .join(monetary, on="customer_id")

rfm.head()
```

### Cohort Analysis

```python
# Cell 4: Cohort retention
# Get first order date per customer (this is the cohort date)
first_order = clean_df.group_by("customer_id").agg({
    "order_date": "min"
})
first_order["cohort_month"] = first_order["order_date"].str.slice(0, 7)

# For complex pivoting, bridge to pandas on the already-aggregated small result
first_order_pd = first_order.to_pandas()[["customer_id", "cohort_month"]]
cohort_pd = clean_df.to_pandas()
cohort_pd = cohort_pd.merge(first_order_pd, on="customer_id")
cohort_pd["order_month"] = cohort_pd["order_date"].str[:7]

cohort_pivot = cohort_pd.groupby(["cohort_month", "order_month"])["customer_id"].nunique()
print(cohort_pivot.head(10))
```

## Notebook 4: Performance Demonstration

### Benchmarking Suite

```python
# Cell 1: Generate test data
import random
import string

def generate_large_dataset(n_rows):
    """Generate synthetic dataset for benchmarking."""
    data = []
    categories = ["Electronics", "Clothing", "Home", "Sports", "Books"]
    regions = ["North", "South", "East", "West"]

    for i in range(n_rows):
        data.append({
            "id": i,
            "product": ''.join(random.choices(string.ascii_uppercase, k=8)),
            "category": random.choice(categories),
            "price": round(random.uniform(10, 500), 2),
            "quantity": random.randint(1, 50),
            "region": random.choice(regions),
            "discount": round(random.uniform(0, 0.3), 2),
        })
    return data

print("Generating 1 million rows...")
data = generate_large_dataset(1_000_000)
print("Done!")
```

```python
# Cell 2: Benchmark DataFrame creation
import time

start = time.time()
df = DataFrame(data)
creation_time = time.time() - start

print(f"Time to create DataFrame: {creation_time:.2f}s")
print(f"Rows per second: {1_000_000 / creation_time:,.0f}")
```

```python
# Cell 3: Benchmark filtering
start = time.time()
filtered = df[df["price"] > 250]
filter_time = time.time() - start

print(f"Filter (price > 250): {filter_time*1000:.1f}ms")
print(f"Result: {filtered.shape[0]:,} rows")
```

```python
# Cell 4: Benchmark aggregation
start = time.time()
grouped = df.group_by("category").agg({
    "price": ["sum", "mean", "min", "max"],
    "quantity": "sum"
})
agg_time = time.time() - start

print(f"GroupBy aggregation: {agg_time*1000:.1f}ms")
print(grouped)
```

```python
# Cell 5: Memory efficiency
import sys

# Estimate memory: our engine uses columnar layout with fixed-width types
n_rows, n_cols = df.shape
# ~8 bytes per value for Int64/Float64; strings vary by length
estimated_mb = (n_rows * n_cols * 8) / (1024 * 1024)
print(f"Memory estimate for {n_rows:,} rows × {n_cols} cols: ~{estimated_mb:.0f} MB")
print(f"  (Actual: less for sparse/string columns, this is an upper bound for fixed-width)")
```

## Interactive Visualization

### Using Matplotlib with Rust DataFrames

```python
# Cell 6: Convert to Pandas for visualization
import matplotlib.pyplot as plt

# Export to Pandas for plotting
pdf = df.to_pandas()

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Price distribution
pdf["price"].hist(ax=axes[0, 0], bins=50)
axes[0, 0].set_title("Price Distribution")

# Quantity by category
pdf.groupby("category")["quantity"].sum().plot.bar(ax=axes[0, 1])
axes[0, 1].set_title("Quantity by Category")

# Revenue by region
pdf["revenue"] = pdf["price"] * pdf["quantity"]
pdf.groupby("region")["revenue"].sum().plot.pie(ax=axes[1, 0])
axes[1, 0].set_title("Revenue by Region")

# Price vs Quantity scatter (sample)
sample = pdf.sample(10000)
axes[1, 1].scatter(sample["price"], sample["quantity"], alpha=0.1)
axes[1, 1].set_title("Price vs Quantity")

plt.tight_layout()
plt.show()
```

## Best Practices for Interactive Work

### 1. Lazy Evaluation Awareness

```python
# Some operations may be lazy - force evaluation
df = read_csv("large_file.csv")

# This triggers actual data loading
_ = df.shape  # or df.head()
```

### 2. Memory Management

```python
# Explicitly delete large DataFrames to trigger Rust memory release
temp_df = read_csv("temp_analysis.csv")
result = temp_df.group_by("category").agg({"value": "sum"})
del temp_df  # Calls __del__ → hf_free_manager() → Rust heap freed immediately
```

### 3. Caching Intermediate Results

```python
# Rust operations are fast enough that intermediate DataFrames
# can be reused directly without explicit caching
filtered_df = df[df["revenue"] > 1000]

# Both operations reuse the already-materialized filtered_df in memory
result1 = filtered_df.group_by("region").agg({"revenue": "sum"})
result2 = filtered_df.group_by("category").agg({"revenue": "mean"})
```

## Troubleshooting

### Common Issues

```python
# Issue: "Null pointer" error
# Cause: Rust operation failed, check input data
try:
    df = read_csv("missing_file.csv")
except RuntimeError as e:
    print(f"Load failed: {e}")

# Issue: Schema mismatch in joins
# Solution: Ensure column types match before joining
df1["key"] = df1["key"].astype("Int64")
df2["key"] = df2["key"].astype("Int64")
result = df1.join(df2, on="key")
```

## Summary

This chapter demonstrated:

- **Interactive Exploration**: Loading, inspecting, and summarizing data in Jupyter
- **Transformation Pipelines**: Step-by-step cleaning and feature engineering
- **Advanced Analytics**: Time series, RFM analysis, cohort analysis
- **Performance Benchmarks**: Quantitative demonstration of Rust speed advantages
- **Visualization**: Integrating with matplotlib for charts
- **Best Practices**: Memory management, caching, and troubleshooting

The hybrid Rust+Python engine provides the performance needed for large datasets while maintaining the interactive, exploratory workflow that data engineers expect from Python.
