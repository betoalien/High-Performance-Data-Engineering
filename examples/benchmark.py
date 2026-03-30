#!/usr/bin/env python3
"""
Performance Benchmark Suite

Compare the hybrid Rust+Python engine against pure Python and Pandas.

Usage:
    python benchmark.py [--rows N]
"""

import argparse
import random
import string
import time
from typing import Callable, Dict, List

# Try to import all comparison targets
try:
    from pardox import DataFrame as RustDataFrame, read_csv
    HAS_RUST = True
except ImportError:
    HAS_RUST = False
    print("Warning: pardox SDK not available - skipping Rust benchmarks")

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("Warning: pandas not available - skipping Pandas benchmarks")


def generate_test_data(n_rows: int) -> List[dict]:
    """Generate synthetic dataset for benchmarking."""
    categories = ["Electronics", "Clothing", "Home", "Sports", "Books"]
    regions = ["North", "South", "East", "West", "Central"]

    data = []
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


def benchmark_dataframe_creation(data: List[dict], name: str,
                                  df_class: Callable) -> float:
    """Benchmark DataFrame creation time."""
    start = time.time()
    df = df_class(data)
    # Force materialization
    _ = df.shape if hasattr(df, 'shape') else len(df)
    elapsed = time.time() - start
    print(f"  {name}: {elapsed:.3f}s")
    return elapsed


def benchmark_filtering(df, name: str) -> float:
    """Benchmark filtering operation."""
    start = time.time()

    if hasattr(df, '__getitem__') and hasattr(df, 'shape'):
        # Rust DataFrame or Pandas
        filtered = df[df["price"] > 250]
        _ = filtered.shape if hasattr(filtered, 'shape') else len(filtered)
    else:
        # Pure Python
        filtered = [r for r in df if r["price"] > 250]

    elapsed = time.time() - start
    result_count = len(filtered) if hasattr(filtered, '__len__') else filtered.shape[0]
    print(f"  {name}: {elapsed:.3f}s ({result_count:,} rows)")
    return elapsed


def benchmark_aggregation(df, name: str) -> float:
    """Benchmark GroupBy aggregation."""
    start = time.time()

    if hasattr(df, 'group_by'):
        # Rust DataFrame
        result = df.group_by("category").agg({"price": ["sum", "mean"]})
    elif hasattr(df, 'groupby'):
        # Pandas
        result = df.groupby("category")["price"].agg(["sum", "mean"])
    else:
        # Pure Python
        from collections import defaultdict
        sums = defaultdict(float)
        counts = defaultdict(int)
        for row in df:
            sums[row["category"]] += row["price"]
            counts[row["category"]] += 1
        result = {cat: {"sum": sums[cat], "mean": sums[cat]/counts[cat]}
                  for cat in sums}

    elapsed = time.time() - start
    print(f"  {name}: {elapsed:.3f}s")
    return elapsed


def run_benchmarks(n_rows: int):
    """Run complete benchmark suite."""
    print("=" * 60)
    print(f"BENCHMARK SUITE - {n_rows:,} rows")
    print("=" * 60)

    # Generate test data
    print("\nGenerating test data...")
    data = generate_test_data(n_rows)
    print(f"Generated {len(data):,} records")

    results: Dict[str, Dict[str, float]] = {
        "creation": {},
        "filtering": {},
        "aggregation": {}
    }

    # === PURE PYTHON (baseline) ===
    print("\n--- Pure Python ---")

    elapsed = benchmark_dataframe_creation(data, "Creation", list)
    results["creation"]["Python"] = elapsed

    elapsed = benchmark_filtering(data, "Filter")
    results["filtering"]["Python"] = elapsed

    elapsed = benchmark_aggregation(data, "Aggregation")
    results["aggregation"]["Python"] = elapsed

    # === PANDAS ===
    if HAS_PANDAS:
        print("\n--- Pandas ---")

        def pandas_df(data):
            return pd.DataFrame(data)

        elapsed = benchmark_dataframe_creation(data, "Creation", pandas_df)
        results["creation"]["Pandas"] = elapsed

        pdf = pd.DataFrame(data)
        elapsed = benchmark_filtering(pdf, "Filter")
        results["filtering"]["Pandas"] = elapsed

        elapsed = benchmark_aggregation(pdf, "Aggregation")
        results["aggregation"]["Pandas"] = elapsed

    # === RUST + PYTHON ===
    if HAS_RUST:
        print("\n--- Rust+Python (Hybrid) ---")

        elapsed = benchmark_dataframe_creation(data, "Creation", RustDataFrame)
        results["creation"]["Rust+Python"] = elapsed

        rdf = RustDataFrame(data)
        elapsed = benchmark_filtering(rdf, "Filter")
        results["filtering"]["Rust+Python"] = elapsed

        elapsed = benchmark_aggregation(rdf, "Aggregation")
        results["aggregation"]["Rust+Python"] = elapsed

    # === SUMMARY ===
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for operation, times in results.items():
        print(f"\n{operation.upper()}:")
        for engine, elapsed in sorted(times.items(), key=lambda x: x[1]):
            print(f"  {engine}: {elapsed:.3f}s")

    # Calculate speedups
    if HAS_RUST:
        print("\n--- Speedup vs Pure Python ---")
        for operation in results:
            python_time = results[operation].get("Python", 0)
            rust_time = results[operation].get("Rust+Python", 0)
            if python_time > 0 and rust_time > 0:
                speedup = python_time / rust_time
                print(f"  {operation}: {speedup:.1f}x faster")

        if HAS_PANDAS:
            print("\n--- Speedup vs Pandas ---")
            for operation in results:
                pandas_time = results[operation].get("Pandas", 0)
                rust_time = results[operation].get("Rust+Python", 0)
                if pandas_time > 0 and rust_time > 0:
                    speedup = pandas_time / rust_time
                    print(f"  {operation}: {speedup:.1f}x faster")


def main():
    parser = argparse.ArgumentParser(description="Performance benchmark suite")
    parser.add_argument(
        "--rows", "-n",
        type=int,
        default=100_000,
        help="Number of rows for benchmark"
    )
    args = parser.parse_args()

    run_benchmarks(args.rows)


if __name__ == "__main__":
    main()
