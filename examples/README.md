# Example Code

This directory contains runnable examples demonstrating the hybrid Rust+Python DataFrame engine.

## Files

### `etl_pipeline.py`

A complete ETL (Extract, Transform, Load) pipeline that:
- Loads data from CSV files using the Rust engine
- Cleans and transforms data with vectorized operations
- Computes aggregations (GroupBy, summaries)
- Writes output to Parquet format

**Usage:**
```bash
python etl_pipeline.py --input data/raw/orders.csv --output data/processed/orders.parquet
```

**Options:**
- `--input, -i`: Input CSV file path (default: `data/raw/orders.csv`)
- `--output, -o`: Output Parquet file path (default: `data/processed/orders.parquet`)
- `--threshold, -t`: Filter threshold for order value (default: 100.0)

### `benchmark.py`

Performance benchmark suite comparing:
- Pure Python (list/dict operations)
- Pandas DataFrame
- Hybrid Rust+Python engine

**Usage:**
```bash
python benchmark.py --rows 100000
```

**Options:**
- `--rows, -n`: Number of rows for benchmark (default: 100,000)

## Requirements

```bash
pip install -e ../pardox_project_sdk/
pip install pandas  # For comparison benchmarks
```

## Expected Output

### ETL Pipeline
```
============================================================
HIGH-PERFORMANCE ETL PIPELINE
============================================================
Input:  data/raw/orders.csv
Output: data/processed/orders.parquet
Filter: total_value > $100.0

=== EXTRACT: Loading data/raw/orders.csv ===
  Loaded 1,000,000 rows in 1.23s
  Columns: ['order_id', 'customer_id', 'product_sku', 'quantity', 'unit_price', 'order_date', 'region']

=== TRANSFORM: Cleaning and enriching ===
  Filling null values...
  Adding derived columns...
  Filtering orders with value > $100.0...
  Transform completed in 0.45s
  Result: 847,293 rows after filtering

=== LOAD: Aggregating and writing output ===
  Computing regional summary...
  Writing to data/processed/orders.parquet...

============================================================
PIPELINE COMPLETED in 2.15s
============================================================
```

### Benchmark
```
============================================================
BENCHMARK SUITE - 100,000 rows
============================================================

--- Pure Python ---
  Creation: 0.089s
  Filter: 0.045s
  Aggregation: 0.156s

--- Pandas ---
  Creation: 0.012s
  Filter: 0.003s
  Aggregation: 0.008s

--- Rust+Python (Hybrid) ---
  Creation: 0.002s
  Filter: 0.0004s
  Aggregation: 0.001s

--- Speedup vs Pure Python ---
  creation: 44.5x faster
  filtering: 112.5x faster
  aggregation: 156.0x faster

--- Speedup vs Pandas ---
  creation: 6.0x faster
  filtering: 7.5x faster
  aggregation: 8.0x faster
```
