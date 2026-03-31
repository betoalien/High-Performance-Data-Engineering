---
title: Chapter 9 - Profiling and Optimization
parent: High-Performance Data Engineering
nav_order: 10
has_children: false
description: Flamegraphs, SIMD analysis, memory profiling, and systematic performance tuning for hybrid Rust+Python engines
---

# Chapter 9: Profiling and Optimization

You have a working hybrid engine. Now the question is: **how fast is it, where does time go, and what can be improved?** Gut feeling is not enough. Profiling gives you data; optimization without profiling is guessing.

The golden rule of performance engineering:

> **Measure first. Optimize second. Measure again.**

---

## Section 1: Profiling the Python Layer

Start with the Python layer. It is the easiest to profile and often the first place time is wasted in setup and orchestration code.

### cProfile: Finding Hot Python Code

`cProfile` is built into Python and has zero dependencies:

```python
# profile_pipeline.py
import cProfile
import pstats
import io
from hyperframe import read_csv

def run_pipeline(csv_path: str):
    df = read_csv(csv_path)
    filtered = df.filter("revenue", ">", 100.0)
    grouped  = filtered.groupby_sum("region", "revenue")
    sorted_  = grouped.sort_by("revenue", ascending=False)
    return sorted_

if __name__ == "__main__":
    pr = cProfile.Profile()
    pr.enable()

    for _ in range(100):
        run_pipeline("data/sales_1m.csv")

    pr.disable()

    stream = io.StringIO()
    ps = pstats.Stats(pr, stream=stream).sort_stats("cumulative")
    ps.print_stats(20)
    print(stream.getvalue())
```

```
ncalls  tottime  percall  cumtime  percall filename:lineno(function)
   100    0.002    0.000    8.312    0.083 profile_pipeline.py:7(run_pipeline)
   100    0.000    0.000    7.891    0.079 io.py:4(read_csv)
   100    0.000    0.000    7.890    0.079 wrapper.py:94(<module>)
   100    0.000    0.000    0.310    0.003 frame.py:44(filter)
   100    0.000    0.000    0.089    0.001 frame.py:68(groupby_sum)
```

What to look for:
- Time in Python code itself (setup, encoding strings, JSON serialization)
- Repeated ctypes calls that could be batched
- Time spent in `json.dumps` for `DataFrame()` constructor (NDJSON serialization)

### py-spy: Low-Overhead Sampling Profiler

`py-spy` profiles a running process without modifying code:

```bash
pip install py-spy

# Profile a script
py-spy record -o profile.svg -- python examples/etl_pipeline.py

# Profile a running process
py-spy record -o profile.svg --pid 12345
```

`py-spy` produces a flamegraph SVG showing which Python frames consume time. Open it in a browser. The width of each bar = fraction of time spent there.

### Key Insight: Most Time Should Be in Rust

For a well-implemented hybrid engine, a `cProfile` run should show:

```
ncalls  tottime  cumtime  function
   100    0.001    8.310  ctypes function: hf_frame_from_csv
   100    0.000    0.310  ctypes function: hf_frame_filter_gt
```

If `tottime` in Python code is significant relative to ctypes calls, the bottleneck is in Python orchestration (string encoding, JSON conversion), not the Rust engine. Common fixes:

- Batch small operations into a single FFI call
- Cache `.encode()` results if the same column name is used repeatedly
- Replace `json.dumps` row-by-row with bulk serialization

---

## Section 2: Profiling the Rust Engine

### `cargo bench`: Micro-Benchmarks with Criterion

Criterion is the standard Rust benchmarking library. It runs each benchmark hundreds of times, applies statistical analysis, and detects regressions.

Add to `Cargo.toml`:

```toml
[dev-dependencies]
criterion = { version = "0.5", features = ["html_reports"] }

[[bench]]
name = "engine_benches"
harness = false
```

```rust
// benches/engine_benches.rs
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use hyperframe_core::{Frame, Column, DType};

fn make_frame(n: usize) -> Frame {
    Frame {
        schema: vec![
            ("price".into(),  DType::Float64),
            ("region".into(), DType::Text),
        ],
        columns: vec![
            Column::Float64((0..n).map(|i| i as f64).collect()),
            Column::Text((0..n).map(|i| {
                if i % 2 == 0 { "North".into() } else { "South".into() }
            }).collect()),
        ],
        nrows: n,
    }
}

fn bench_sum(c: &mut Criterion) {
    let mut group = c.benchmark_group("sum");

    for size in [100_000, 1_000_000, 10_000_000] {
        let frame = make_frame(size);
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            &frame,
            |b, f| b.iter(|| black_box(f.sum("price").unwrap())),
        );
    }
    group.finish();
}

fn bench_filter(c: &mut Criterion) {
    let mut group = c.benchmark_group("filter_gt");

    for size in [100_000, 1_000_000] {
        let frame = make_frame(size);
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            &frame,
            |b, f| b.iter(|| black_box(f.filter_gt("price", 50_000.0).unwrap())),
        );
    }
    group.finish();
}

fn bench_groupby(c: &mut Criterion) {
    let frame = make_frame(1_000_000);
    c.bench_function("groupby_sum_1m", |b| {
        b.iter(|| black_box(frame.groupby_sum("region", "price").unwrap()))
    });
}

criterion_group!(benches, bench_sum, bench_filter, bench_groupby);
criterion_main!(benches);
```

Run:
```bash
cargo bench
# HTML report: target/criterion/report/index.html
```

Sample output:
```
sum/100000              time:   [52.3 µs 52.7 µs 53.1 µs]
sum/1000000             time:   [521 µs  524 µs  528 µs]
sum/10000000            time:   [5.21 ms 5.24 ms 5.27 ms]
filter_gt/100000        time:   [31.2 µs 31.5 µs 31.8 µs]
filter_gt/1000000       time:   [312 µs  315 µs  318 µs]
groupby_sum_1m          time:   [24.8 ms 25.1 ms 25.4 ms]
```

This tells you:
- `sum` scales linearly (0.5 µs/element -- cache-limited)
- `filter_gt` is faster than sum (parallel, early exit)
- `groupby_sum` is slower (sequential BTreeMap insertion -- see Chapter 5 for the parallel fix)

---

## Section 3: Flamegraphs with `cargo-flamegraph`

Criterion tells you *how long* operations take. Flamegraphs tell you *why*. A flamegraph shows the call stack at every sample point, making hot code paths instantly visible.

```bash
# Install
cargo install flamegraph
# On Linux, also: sudo apt install linux-perf  (or equivalent)

# Generate flamegraph for a benchmark
cargo flamegraph --bench engine_benches -- --bench bench_sum
# Output: flamegraph.svg
```

Open `flamegraph.svg` in a browser:

```
[frame.sum()]───────────────────────────────────────────────────
  [rayon::par_iter.sum()]──────────────────────────────────────
    [rayon::iter::sum_op]─────────────────────────────
      [__memcpy_avx_unaligned]──────  [__memmove_avx]──────
```

What to look for in a flamegraph:
- **Wide bars**: Functions that consume significant time
- **Tall stacks**: Deep call chains that could be simplified
- **System library calls**: `memcpy`, `malloc` appearing prominently suggests excessive copying

### Reading a Typical Flamegraph for Our Engine

For `groupby_sum` on 1M rows:

```
[groupby_sum()]──────────────────────────────────────────────────  100%
  [BTreeMap::insert()]──────────────────────────────────────────   82%
    [BTreeMap::rebalance()]────────────────────────────────────    71%
  [Column::Text iter]──────────────────────────────────────────    12%
  [Column::Float64 iter]────────────────────────────────────────    6%
```

The flamegraph immediately shows that 82% of time is in `BTreeMap::insert` with rebalancing. This is why Chapter 5 replaces BTreeMap with DashMap for the parallel version -- hash map insertion is O(1) instead of O(log n), and the map itself can be sharded across threads.

---

## Section 4: SIMD Analysis

SIMD (Single Instruction Multiple Data) allows one CPU instruction to process 4–8 floating point values simultaneously. Rust's compiler auto-vectorizes many tight loops, but it is not guaranteed.

### Checking Auto-Vectorization

Use Godbolt Compiler Explorer (compiler-explorer.com) or local assembly output:

```bash
# Generate annotated assembly
RUSTFLAGS="-C target-cpu=native" cargo rustc --release -- --emit=asm
# Assembly in: target/release/deps/hyperframe_core-*.s
```

Look for vectorized instructions in your sum loop:

```asm
# Scalar (no SIMD):
.LBB0_4:
    addsd   xmm0, qword ptr [rsi + 8*rcx]   ; adds one f64
    inc     rcx
    cmp     rcx, rdx
    jl      .LBB0_4

# AVX2 vectorized (4 f64 per instruction):
.LBB0_8:
    vaddpd  ymm0, ymm0, ymmword ptr [rsi + rcx]   ; adds 4 f64
    vaddpd  ymm1, ymm1, ymmword ptr [rsi + rcx + 32]
    add     rcx, 64
    cmp     rcx, rdi
    jl      .LBB0_8
```

`vaddpd ymm0` processes 4 doubles in one instruction -- 4x throughput over scalar.

### Enabling AVX2 / AVX-512

```toml
# Cargo.toml
[profile.release]
opt-level = 3
lto = "fat"
codegen-units = 1
```

```bash
# Build for the current CPU (enables all available SIMD)
RUSTFLAGS="-C target-cpu=native" cargo build --release

# Or specify explicitly:
RUSTFLAGS="-C target-feature=+avx2" cargo build --release
RUSTFLAGS="-C target-feature=+avx512f" cargo build --release
```

Note: Binaries compiled with `target-cpu=native` only run on CPUs with equivalent or better feature sets. For distribution, compile for a baseline (e.g., `x86-64-v3` for AVX2 support on modern x86).

### SIMD Speedups Observed

| Operation | Scalar | AVX2 | AVX-512 | Speedup |
|-----------|--------|------|---------|---------|
| `sum` (f64) | 1.0x | 3.8x | 7.2x | Cache-limited at large sizes |
| `filter_gt` | 1.0x | 3.5x | 6.8x | Branch prediction also matters |
| `mean` | 1.0x | 3.9x | 7.5x | Pure reduction, SIMD ideal |

---

## Section 5: Memory Profiling

Memory efficiency is as important as speed. A pipeline that uses 65% less memory than pandas can process larger datasets without out-of-memory errors.

### Tracking Rust Allocations with DHAT

DHAT is a Valgrind tool that tracks every heap allocation:

```bash
cargo install cargo-valgrind   # or use system valgrind
valgrind --tool=dhat --dhat-out-file=dhat.out \
    python -c "
from hyperframe import read_csv
df = read_csv('data/sales_10m.csv')
print(df.sum('revenue'))
"
dh_view.py dhat.out
```

What to look for:
- Total heap bytes at peak
- Allocations per operation (excessive allocations indicate unnecessary copying)
- Long-lived vs short-lived allocations

### Tracking Python Memory

```python
import tracemalloc
from hyperframe import read_csv

tracemalloc.start()

df = read_csv("data/sales_10m.csv")
total = df.sum("revenue")
filtered = df.filter("revenue", ">", 1000.0)

snapshot = tracemalloc.take_snapshot()
top = snapshot.statistics("lineno")[:10]
for stat in top:
    print(stat)
```

For a hybrid engine, Python memory should be minimal:
```
hyperframe/frame.py:25: size=56 B, count=3, average=19 B   # _ptr fields
hyperframe/io.py:8:    size=1024 B, count=1                 # path encoding
```

All the data lives in Rust's heap, invisible to `tracemalloc`.

### Comparing Memory Usage: hyperframe vs pandas

```python
import psutil
import os
import gc
from hyperframe import read_csv

def current_rss_mb():
    """Resident Set Size in megabytes."""
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024

# Measure hyperframe
baseline = current_rss_mb()
df_hf = read_csv("data/sales_10m.csv")
rss_hyperframe = current_rss_mb() - baseline
print(f"hyperframe: {rss_hyperframe:.1f} MB")

del df_hf
gc.collect()

# Measure pandas
import pandas as pd
baseline = current_rss_mb()
df_pd = pd.read_csv("data/sales_10m.csv")
rss_pandas = current_rss_mb() - baseline
print(f"pandas:     {rss_pandas:.1f} MB")

print(f"Reduction:  {(1 - rss_hyperframe / rss_pandas) * 100:.1f}%")
```

Typical output for a 10M row, 5 column mixed-type CSV:

```
hyperframe: 412 MB    (columnar, no object overhead)
pandas:     1140 MB   (object dtype + Python overhead)
Reduction:  63.8%
```

Why the difference:
- Pandas uses Python objects for string columns (56 bytes/object on CPython)
- HyperFrame stores strings as `Vec<String>` with compact heap layout
- Pandas keeps index, dtypes metadata, and NumPy arrays in addition to data

---

## Section 6: End-to-End Benchmark

A complete benchmark comparing hyperframe against pandas across the full pipeline:

```python
# examples/benchmark.py (abridged)
import time
import gc
import argparse
import random
import string
import tempfile
import os

def generate_csv(path: str, rows: int):
    regions = ["North", "South", "East", "West"]
    with open(path, "w") as f:
        f.write("id,region,revenue,qty,active\n")
        for i in range(rows):
            region = regions[i % 4]
            revenue = round(random.uniform(10.0, 10_000.0), 2)
            qty = random.randint(1, 100)
            active = "true" if random.random() > 0.3 else "false"
            f.write(f"{i},{region},{revenue},{qty},{active}\n")

def benchmark_hyperframe(csv_path: str):
    from hyperframe import read_csv
    results = {}

    t0 = time.perf_counter()
    df = read_csv(csv_path)
    results["csv_load"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    filtered = df.filter("revenue", ">", 5000.0)
    results["filter"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    grouped = df.groupby_sum("region", "revenue")
    results["groupby"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    _ = df.sum("revenue")
    results["sum"] = time.perf_counter() - t0

    return results

def benchmark_pandas(csv_path: str):
    import pandas as pd
    results = {}

    t0 = time.perf_counter()
    df = pd.read_csv(csv_path)
    results["csv_load"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    filtered = df[df["revenue"] > 5000.0]
    results["filter"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    grouped = df.groupby("region")["revenue"].sum()
    results["groupby"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    _ = df["revenue"].sum()
    results["sum"] = time.perf_counter() - t0

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=1_000_000)
    args = parser.parse_args()

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        csv_path = f.name

    try:
        print(f"Generating {args.rows:,} row CSV...")
        generate_csv(csv_path, args.rows)

        print("\nRunning hyperframe benchmark...")
        hf = benchmark_hyperframe(csv_path)

        gc.collect()

        print("Running pandas benchmark...")
        pd_ = benchmark_pandas(csv_path)

        print(f"\n{'Operation':<15} {'hyperframe':>12} {'pandas':>12} {'Speedup':>10}")
        print("-" * 52)
        for op in ["csv_load", "filter", "groupby", "sum"]:
            speedup = pd_[op] / hf[op]
            print(f"{op:<15} {hf[op]*1000:>10.1f}ms {pd_[op]*1000:>10.1f}ms "
                  f"{speedup:>8.1f}x")
    finally:
        os.unlink(csv_path)
```

Sample output for 1M rows:

```
Operation       hyperframe       pandas    Speedup
----------------------------------------------------
csv_load          187.3ms       1243.5ms     6.6x
filter             12.1ms         98.4ms     8.1x
groupby            24.8ms        128.9ms     5.2x
sum                 5.2ms         43.7ms     8.4x
```

---

## Section 7: Systematic Optimization Workflow

Use this checklist when trying to improve performance:

### Step 1: Profile before changing anything

```bash
cargo bench > before.txt
py-spy record -o before.svg -- python benchmark.py
```

### Step 2: Identify the bottleneck

From the flamegraph and benchmarks, rank bottlenecks:
1. Is the bottleneck in Rust or Python? (If in Python: likely setup/encoding overhead)
2. Is it CPU-bound (compute) or memory-bound (cache pressure)?
3. Is it sequential (algorithmic improvement needed) or parallel (Rayon)?

### Step 3: Apply one change at a time

Common optimizations in order of impact:

| Change | Typical Speedup | Risk |
|--------|-----------------|------|
| `par_iter()` over `iter()` (Chapter 5) | 4–8x (multi-core) | Race conditions if done wrong |
| `target-cpu=native` SIMD | 2–4x | Binary incompatibility |
| `BTreeMap` → `DashMap` for groupby | 3–5x | Extra dependency |
| `lto="fat"` in Cargo.toml | 5–15% | Longer compile time |
| Reduce string allocations in hot paths | 10–30% | Code complexity |
| Replace NDJSON with binary format | 2–10x for constructor | Schema complexity |

### Step 4: Measure again

```bash
cargo bench > after.txt
diff before.txt after.txt
```

Criterion includes change detection -- it will report:
```
sum/1000000: change: [-23.4% -21.2% -19.0%] (p = 0.00 < 0.05)
             Performance has improved.
```

If the improvement is not statistically significant (`p > 0.05`), it may be noise.

---

## Summary

In this chapter you learned to systematically profile and optimize a hybrid Rust+Python engine:

- **cProfile / py-spy**: Find Python-layer overhead; confirm most time is in ctypes calls
- **cargo bench + Criterion**: Micro-benchmark individual Rust functions with statistical rigor
- **Flamegraphs**: Visualize hot code paths and identify algorithmic bottlenecks
- **SIMD analysis**: Confirm auto-vectorization is occurring; enable `target-cpu=native`
- **Memory profiling**: Compare heap usage between hyperframe and pandas
- **Systematic workflow**: Profile → identify → change one thing → measure again

The key insight: most time in a well-built hybrid engine should be spent doing useful computation in Rust, not in Python orchestration or FFI overhead. If `cProfile` shows significant time in Python code, the Python layer needs optimization. If the Rust flamegraph shows time in memory allocation rather than compute, the algorithm may need structural changes.

---

**Next**: [Chapter 10 - Interactive Data Engineering](10-jupyter-lab.md) -- using Jupyter notebooks for interactive exploration with the hybrid engine.
