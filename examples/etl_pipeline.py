#!/usr/bin/env python3
"""
Complete ETL Pipeline Example

This script demonstrates a production-ready ETL pipeline using the
hybrid Rust+Python DataFrame engine.

Usage:
    python etl_pipeline.py [--input INPUT] [--output OUTPUT]
"""

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

# Import from the hybrid SDK
try:
    from pardox import DataFrame, read_csv
except ImportError:
    print("Error: pardox SDK not installed.")
    print("Run: pip install -e pardox_project_sdk/")
    raise


@dataclass
class PipelineConfig:
    """Configuration for the ETL pipeline."""
    input_path: str = "data/raw/orders.csv"
    output_path: str = "data/processed/orders.parquet"
    summary_path: str = "data/processed/summary.csv"
    filter_threshold: float = 100.0
    null_quantity_default: int = 1
    null_price_default: float = 0.0


def extract_orders(csv_path: str) -> DataFrame:
    """
    EXTRACT: Load orders from CSV file.

    The Rust engine handles:
    - Parallel file reading with multiple threads
    - Automatic type inference
    - Memory-efficient chunked loading
    """
    print(f"\n=== EXTRACT: Loading {csv_path} ===")
    start = time.time()

    # Optional schema for faster loading and type safety
    schema = {
        "order_id": "Utf8",
        "customer_id": "Utf8",
        "product_sku": "Utf8",
        "quantity": "Int64",
        "unit_price": "Float64",
        "order_date": "Utf8",
        "region": "Utf8"
    }

    df = read_csv(csv_path, schema=schema)
    elapsed = time.time() - start

    print(f"  Loaded {df.shape[0]:,} rows in {elapsed:.2f}s")
    print(f"  Columns: {df.columns}")

    return df


def transform_data(df: DataFrame) -> DataFrame:
    """
    TRANSFORM: Clean and enrich the data.

    Operations:
    - Handle null values
    - Add derived columns
    - Filter low-value orders
    """
    print("\n=== TRANSFORM: Cleaning and enriching ===")
    start = time.time()

    # Handle null values
    print("  Filling null values...")
    df.fill_na("quantity", PipelineConfig.null_quantity_default)
    df.fill_na("unit_price", PipelineConfig.null_price_default)

    # Add derived columns (vectorized arithmetic in Rust)
    print("  Adding derived columns...")
    df["total_value"] = df["quantity"] * df["unit_price"]
    df["order_month"] = df["order_date"].str.slice(0, 7)  # "YYYY-MM"

    # Filter to meaningful orders
    print(f"  Filtering orders with value > ${PipelineConfig.filter_threshold}...")
    df = df[df["total_value"] > PipelineConfig.filter_threshold]

    elapsed = time.time() - start
    print(f"  Transform completed in {elapsed:.2f}s")
    print(f"  Result: {df.shape[0]:,} rows after filtering")

    return df


def load_data(df: DataFrame, config: PipelineConfig):
    """
    LOAD: Aggregate and write output files.
    """
    print("\n=== LOAD: Aggregating and writing output ===")
    start = time.time()

    # Aggregate by region
    print("  Computing regional summary...")
    regional = df.group_by("region").agg({
        "total_value": ["sum", "mean"],
        "order_id": "count"
    })
    print("\n  Regional Summary:")
    print(regional)

    # Aggregate by month (time series)
    print("\n  Computing monthly trends...")
    monthly = df.group_by("order_month").agg({
        "total_value": "sum",
        "order_id": "count"
    }).sort_by("order_month")

    # Write outputs
    print(f"\n  Writing to {config.output_path}...")
    df.to_parquet(
        config.output_path,
        compression="snappy",
        row_group_size=100_000
    )

    print(f"  Writing summary to {config.summary_path}...")
    regional.to_pandas().to_csv(config.summary_path, index=False)

    elapsed = time.time() - start
    print(f"\n  LOAD completed in {elapsed:.2f}s")

    return regional, monthly


def run_pipeline(config: PipelineConfig) -> tuple:
    """
    Execute the complete ETL pipeline.

    Returns:
        tuple: (processed_df, regional_summary, monthly_trends)
    """
    print("=" * 60)
    print("HIGH-PERFORMANCE ETL PIPELINE")
    print("=" * 60)
    print(f"Input:  {config.input_path}")
    print(f"Output: {config.output_path}")
    print(f"Filter: total_value > ${config.filter_threshold}")

    total_start = time.time()

    # Execute pipeline stages
    df = extract_orders(config.input_path)
    df = transform_data(df)
    regional, monthly = load_data(df, config)

    total_elapsed = time.time() - total_start

    print("\n" + "=" * 60)
    print(f"PIPELINE COMPLETED in {total_elapsed:.2f}s")
    print("=" * 60)

    return df, regional, monthly


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="High-performance ETL pipeline with Rust+Python"
    )
    parser.add_argument(
        "--input", "-i",
        default="data/raw/orders.csv",
        help="Input CSV file path"
    )
    parser.add_argument(
        "--output", "-o",
        default="data/processed/orders.parquet",
        help="Output Parquet file path"
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=100.0,
        help="Filter threshold for order value"
    )

    args = parser.parse_args()

    config = PipelineConfig(
        input_path=args.input,
        output_path=args.output,
        filter_threshold=args.threshold
    )

    run_pipeline(config)


if __name__ == "__main__":
    main()
