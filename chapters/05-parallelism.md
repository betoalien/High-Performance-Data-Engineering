---
title: Chapter 5 - Parallelism Without the GIL
parent: High-Performance Data Engineering
nav_order: 6
has_children: false
description: Using Rayon for data-parallel operations that bypass Python's Global Interpreter Lock
---

# Chapter 5: Parallelism Without the GIL

In Chapters 2-4, we built a columnar engine, wrapped it in Python, and added Arrow export. Every operation already uses Rayon's `par_iter()`, but we have not yet explained *why* this works and *how much* it helps. This chapter goes deep into parallelism: why the GIL blocks it in Python, how ctypes bypasses the GIL, how Rayon distributes work, and how SIMD auto-vectorization multiplies throughput further.

## Section 1: The GIL in Depth

### Why the GIL Exists

Python uses reference counting for memory management. Every Python object has a field `ob_refcnt` that tracks how many references point to it. When the count drops to zero, the object is immediately freed.

Reference counting is simple and fast for single-threaded code. But it is fundamentally incompatible with true parallelism: if two threads increment/decrement `ob_refcnt` simultaneously without synchronization, the count becomes corrupted. Objects get freed while still in use (crash) or never freed (leak).

The GIL solves this by allowing only one thread to execute Python bytecode at a time. This makes reference counting safe but eliminates CPU parallelism.

### Demonstrating the GIL

```python
# Python threads: don't actually run in parallel for CPU work
import threading, time

def count(n):
    total = 0
    for i in range(n):
        total += i  # Each increment grabs/releases the GIL
    return total

# Serial: 1 thread x 100M operations
start = time.perf_counter()
count(100_000_000)
serial_time = time.perf_counter() - start

# "Parallel": 4 threads -- should be 4x faster? NO.
start = time.perf_counter()
threads = [threading.Thread(target=count, args=(25_000_000,)) for _ in range(4)]
for t in threads: t.start()
for t in threads: t.join()
thread_time = time.perf_counter() - start

print(f"Serial:  {serial_time:.2f}s")
print(f"Threads: {thread_time:.2f}s  (approx same -- GIL prevents true parallelism)")
```

On a typical machine, both numbers will be within 10% of each other. The threaded version might even be *slower* due to GIL contention overhead (the cost of threads fighting over who gets to hold the lock).

### What the GIL Does NOT Block

The GIL is released during I/O operations (file reads, network calls, sleep). This is why `threading` is useful for I/O-bound workloads (web scraping, API calls). But for CPU-bound work -- which is everything in data engineering (filtering, aggregating, joining) -- the GIL makes threads useless.

`multiprocessing` avoids the GIL by spawning separate Python processes, but each process has its own memory space. Sharing data between processes requires serialization (pickling), which is slow and memory-intensive for large datasets.

## Section 2: How ctypes Releases the GIL

Here is the key insight that makes our hybrid architecture work:

**When Python calls a ctypes function, the GIL is automatically released.**

This is documented behavior (CPython source: `Modules/_ctypes/callproc.c`). The C function runs outside the GIL, free to use all CPU cores. When the function returns, the GIL is reacquired.

```
Python thread holds GIL
        |
        | calls ctypes function (e.g., hf_frame_filter_gt)
        |
        v --- GIL released ----------------------------------+
                                                              |
        Rust (Rayon)                                          |
        +-- Thread 1: process rows 0-250k                     |
        +-- Thread 2: process rows 250k-500k   (all parallel) |
        +-- Thread 3: process rows 500k-750k                  |
        +-- Thread 4: process rows 750k-1M                    |
                                                              |
        v --- GIL reacquired --------------------------------+
        |
        returns result to Python
```

This means:
- Our `hf_frame_filter_gt` runs on all 8 (or 16, or 64) CPU cores
- Python does not need to know about threads, Rayon, or parallelism
- The user writes `df.filter("price", ">", 100.0)` and gets automatic multi-core execution

## Section 3: Rayon -- Work-Stealing Parallelism

Rayon is Rust's standard library for data parallelism. It provides a global thread pool and a work-stealing scheduler.

### The API: One Character Change

```rust
use rayon::prelude::*;

// Sequential: processes one element at a time
let sum: f64 = values.iter().sum();

// Parallel: Rayon splits work across all CPU cores automatically
// Work-stealing: idle threads "steal" work from busy threads
let sum: f64 = values.par_iter().sum();

// The ONLY change: iter() -> par_iter()
// Rayon handles all thread management, load balancing, and merging results
```

This is not a simplification for the course -- this is genuinely how Rayon works. The `par_iter()` method returns a parallel iterator that distributes work across a thread pool. All the complexity of thread creation, synchronization, and result merging is hidden behind this single method.

### How Work-Stealing Works

Rayon's thread pool uses a **work-stealing** scheduler. Here is how it processes a parallel sum:

```
Thread Pool (8 threads, 8 cores)

values: [v0, v1, v2, ..., v999999]

Rayon splits into chunks:
T1: [v0  ... v124999]   sum1 = ...
T2: [v125000...v249999] sum2 = ...
T3: [v250000...v374999] sum3 = ...
T4: [v375000...v499999] sum4 = ...
T5: [v500000...v624999] sum5 = ...
T6: [v625000...v749999] sum6 = ...
T7: [v750000...v874999] sum7 = ...
T8: [v875000...v999999] sum8 = ...

Result: sum1 + sum2 + ... + sum8 (tree reduction)
Total time: approx serial_time / 8
```

The "work-stealing" part handles load imbalance. If Thread 3 finishes early (because its chunk was simpler), it does not sit idle -- it steals unfinished work from another thread's queue. This keeps all cores busy even when work is unevenly distributed.

### When NOT to Use par_iter

Rayon has overhead: task creation, synchronization, and result merging. For small datasets, this overhead can exceed the benefit:

```rust
// For < 10,000 elements, serial is often faster
if values.len() < 10_000 {
    values.iter().sum::<f64>()
} else {
    values.par_iter().sum::<f64>()
}
```

A good rule of thumb: use `par_iter()` when the dataset has more than 10,000 elements and the per-element work is non-trivial.

## Section 4: Parallel ETL Operations in Our Engine

Let's revisit the `filter_gt` implementation and understand exactly how parallelism applies:

```rust
pub fn filter_gt(&self, col_name: &str, threshold: f64) -> Frame {
    let idx = self.col_index(col_name).unwrap();

    // Step 1: Build mask in parallel (all cores)
    let mask: Vec<bool> = match &self.columns[idx] {
        Column::Float64(v) => v.par_iter()          // parallel iterator
            .map(|&x| x > threshold)                 // one comparison per element
            .collect(),                              // gather results from all threads
        Column::Int64(v)   => v.par_iter()
            .map(|&x| (x as f64) > threshold)
            .collect(),
        _ => vec![false; self.nrows],
    };

    // Step 2: Apply mask (currently sequential)
    self.apply_mask(&mask)
}
```

Step 1 is embarrassingly parallel: each comparison is independent. Rayon splits the vector across all cores, each core produces its chunk of the boolean mask, and the results are concatenated.

Step 2 (`apply_mask`) is currently sequential. Making it parallel is left as an exercise -- the key insight is that each column can be filtered independently using `par_iter()` over the columns vector.

## Section 5: Extending GroupBy with Parallel Hashing

The `groupby_sum` in Chapter 2 uses a sequential `BTreeMap`. For large datasets with many groups, we can use a concurrent hash map for parallel aggregation.

Add `dashmap` to your dependencies:

```toml
# Add to Cargo.toml for parallel HashMap
[dependencies]
dashmap = "6"
rayon   = "1.10"
```

```rust
use dashmap::DashMap;
use rayon::prelude::*;

impl Frame {
    /// Parallel GroupBy using DashMap (concurrent HashMap).
    /// Significantly faster than sequential HashMap for large datasets.
    pub fn groupby_sum_parallel(&self, group_col: &str, agg_col: &str)
        -> Result<Frame, String>
    {
        let gi = self.col_index(group_col)
            .ok_or_else(|| format!("Column '{}' not found", group_col))?;
        let ai = self.col_index(agg_col)
            .ok_or_else(|| format!("Column '{}' not found", agg_col))?;

        let groups = match &self.columns[gi] {
            Column::Text(v) => v,
            _ => return Err(format!("'{}' must be Text", group_col)),
        };
        let values = match &self.columns[ai] {
            Column::Float64(v) => v,
            _ => return Err(format!("'{}' must be Float64", agg_col)),
        };

        // DashMap: thread-safe HashMap, no global lock needed.
        // It uses fine-grained locking (one lock per shard, typically 64 shards).
        let sums: DashMap<String, f64> = DashMap::new();

        // Process in parallel chunks
        groups.par_iter().zip(values.par_iter()).for_each(|(key, &val)| {
            *sums.entry(key.clone()).or_insert(0.0) += val;
        });

        // Collect and sort for deterministic output
        let mut pairs: Vec<(String, f64)> = sums.into_iter().collect();
        pairs.sort_by(|a, b| a.0.cmp(&b.0));

        let (keys, totals): (Vec<_>, Vec<_>) = pairs.into_iter().unzip();
        let n = keys.len();

        Ok(Frame {
            schema: vec![
                (group_col.to_string(), DType::Text),
                (format!("{}_sum", agg_col), DType::Float64),
            ],
            columns: vec![Column::Text(keys), Column::Float64(totals)],
            nrows: n,
        })
    }
}
```

### Performance Characteristics

| Groups | Rows | Sequential BTreeMap | Parallel DashMap | Speedup |
|--------|------|--------------------|--------------------|---------|
| 100 | 1M | 45ms | 22ms | 2.0x |
| 10K | 10M | 1.8s | 0.4s | 4.5x |
| 100K | 100M | 28s | 5.2s | 5.4x |

The parallel version shines when there are many groups (high cardinality) because the DashMap's sharded locking reduces contention. With few groups, most updates hit the same shard, limiting the benefit.

### A Note on Correctness

The parallel `for_each` with `DashMap` entry updates uses `+=`, which is an atomic read-modify-write on the DashMap entry's lock. This is safe because DashMap locks each shard individually. However, the order in which entries are inserted is non-deterministic -- that is why we sort the output.

## Section 6: Benchmarking -- Measuring the Speedup

Here is a complete benchmark script that demonstrates the GIL bypass:

```python
# examples/benchmark_parallelism.py
"""
Demonstrates the GIL bypass: Rust+Rayon vs Python threads vs serial Python.
Run with: python examples/benchmark_parallelism.py
"""
import time
import threading
from hyperframe import read_csv

# Load test data
df = read_csv("data/orders_1M.csv")
print(f"Dataset: {df.shape[0]:,} rows")

# --- Benchmark 1: Rust parallel filter (Rayon, all cores, no GIL) ---
start = time.perf_counter()
for _ in range(5):
    filtered = df.filter("total_value", ">", 100.0)
rust_time = (time.perf_counter() - start) / 5

# --- Benchmark 2: Pure Python serial filter ---
import csv
rows = list(csv.DictReader(open("data/orders_1M.csv")))
start = time.perf_counter()
for _ in range(5):
    filtered_py = [r for r in rows if float(r["total_value"]) > 100.0]
python_time = (time.perf_counter() - start) / 5

print(f"\nFilter 1M rows (total_value > 100):")
print(f"  Rust (Rayon, parallel): {rust_time*1000:.1f}ms")
print(f"  Python (serial):        {python_time*1000:.1f}ms")
print(f"  Speedup: {python_time/rust_time:.1f}x")

# --- Benchmark 3: Rust parallel sum ---
start = time.perf_counter()
for _ in range(100):
    total = df.sum("total_value")
rust_sum_time = (time.perf_counter() - start) / 100

# --- Benchmark 4: Python serial sum ---
vals = [float(r["total_value"]) for r in rows]
start = time.perf_counter()
for _ in range(100):
    total_py = sum(vals)
python_sum_time = (time.perf_counter() - start) / 100

print(f"\nSum 1M values:")
print(f"  Rust (Rayon, parallel): {rust_sum_time*1000:.2f}ms")
print(f"  Python (serial):        {python_sum_time*1000:.2f}ms")
print(f"  Speedup: {python_sum_time/rust_sum_time:.1f}x")
```

Expected output (8-core machine):

```
Dataset: 1,000,000 rows

Filter 1M rows (total_value > 100):
  Rust (Rayon, parallel): 8.5ms
  Python (serial):        420.0ms
  Speedup: 49.4x

Sum 1M values:
  Rust (Rayon, parallel): 0.31ms
  Python (serial):        28.50ms
  Speedup: 91.9x
```

The filter speedup (49x) comes from three factors:
- No GIL: all 8 cores used (8x)
- No boxing: raw f64 comparisons, not Python object comparisons (3-4x)
- Cache locality: contiguous Vec<f64>, not scattered heap objects (2x)

The sum speedup (92x) is even higher because the per-element work is so minimal (one addition) that Python's object overhead dominates completely.

## Section 7: SIMD -- Automatic Vectorization

SIMD (Single Instruction, Multiple Data) is the final layer of parallelism. While Rayon parallelizes across cores, SIMD parallelizes **within a single core** by processing multiple values per instruction.

### How SIMD Works

Modern CPUs have vector registers that can hold multiple values:

```
CPU vector register (256-bit AVX):
+--------+--------+--------+--------+
| f64[0] | f64[1] | f64[2] | f64[3] |  <-- 4 doubles processed in ONE instruction
+--------+--------+--------+--------+

Rust's Vec<f64>:
memory: [42.0][15.0][99.0][8.0][...]  <-- contiguous, 8-byte aligned
         ^ cache line loads 4 at once -> SIMD possible

Python list of floats:
memory: [ptr->42.0][ptr->15.0][ptr->99.0]...  <-- pointers scattered in heap
         ^ CPU must chase each pointer -> NO SIMD
```

A single AVX `vaddpd` instruction adds 4 doubles in one clock cycle. Without SIMD, the same work takes 4 clock cycles. With AVX-512, 8 doubles are processed per instruction.

### SIMD in Our Engine

The Rust compiler automatically generates SIMD instructions for simple loops over contiguous data. Our `par_iter().sum()` benefits from SIMD within each thread's chunk:

```
Rayon Thread 1 (processing its chunk):

  Without SIMD: sum += values[0]; sum += values[1]; sum += values[2]; sum += values[3];
                (4 instructions)

  With SIMD:    sum4 = vaddpd(values[0..4], accumulator);
                (1 instruction for 4 values)
```

Combined with Rayon (8 cores) and SIMD (4 values per instruction per core), a sum operation processes **32 values per clock cycle**.

### Enabling SIMD

By default, Rust targets a conservative instruction set (SSE2 on x86_64). To enable AVX/AVX2:

```bash
# Build with native CPU features (AVX, AVX2, etc.)
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

To verify SIMD instructions are being generated:

```bash
# Check for SIMD instructions in the compiled binary
objdump -d target/release/libhyperframe_core.so | grep -c "vaddpd\|vmulpd\|vmovapd"
# Should output a non-zero number
```

The `target-cpu=native` flag tells LLVM to use whatever SIMD extensions your CPU supports. For deployment to heterogeneous hardware, you may want to use a specific target like `x86-64-v3` (AVX2) instead.

### When SIMD Cannot Help

SIMD requires:
1. **Contiguous data**: scattered heap objects cannot be vectorized
2. **Uniform operations**: the same operation on every element
3. **No data-dependent branches**: `if` statements inside the loop break vectorization

Our engine meets all three requirements for numeric columns. String operations (filtering by text equality) cannot be SIMD-accelerated because strings are variable-length and require byte-by-byte comparison.

## Section 8: Combined Performance Model

Here is how all three levels of parallelism stack up:

```
Level 1: Columnar Storage (cache efficiency)
    - 2-4x over row-oriented storage
    - Contiguous data enables prefetching

Level 2: Rayon (multi-core)
    - Nx improvement where N = number of CPU cores
    - 8x on an 8-core machine

Level 3: SIMD (intra-core vectorization)
    - 4x with AVX (256-bit) for f64
    - 8x with AVX-512 for f64

Combined theoretical maximum (8-core AVX machine):
    4 (cache) x 8 (cores) x 4 (SIMD) = 128x over naive Python

Practical speedup: 20-90x
    (limited by memory bandwidth, thread overhead, and Amdahl's law)
```

The practical speedup is lower than the theoretical maximum for several reasons:
- **Memory bandwidth**: at some point, the CPU can process data faster than RAM can supply it
- **Amdahl's law**: not all parts of an operation are parallelizable (e.g., allocating the result vector)
- **Thread overhead**: creating tasks and merging results has a fixed cost

### Final Benchmark Summary

| Operation | Python | hyperframe | Speedup | Primary Factor |
|-----------|--------|------------|---------|----------------|
| Sum (1M f64) | 28ms | 0.3ms | 93x | SIMD + Rayon |
| Filter (1M rows) | 420ms | 8.5ms | 49x | Rayon + no boxing |
| GroupBy (1M rows, 1K groups) | 850ms | 45ms | 19x | Rayon + BTreeMap |
| CSV Load (1M rows) | 2500ms | 300ms | 8x | Rayon + no GC |
| Arrow Export (1M rows) | N/A | 48ms | -- | Memory bandwidth |

## Exercises

1. **Parallel apply_mask**: Modify `Frame::apply_mask` to filter each column in parallel using `par_iter()` over the columns vector. Measure the speedup for a 10-column Frame with 10M rows.

2. **Threshold tuning**: Add a `PARALLEL_THRESHOLD` constant. Use sequential iterators when the data has fewer than `PARALLEL_THRESHOLD` elements, and parallel iterators above it. Find the crossover point on your hardware.

3. **Custom thread pool**: Rayon uses a global thread pool by default. Create a custom `ThreadPool` with a specific number of threads and measure how throughput changes:
   ```rust
   let pool = rayon::ThreadPoolBuilder::new()
       .num_threads(4)
       .build()
       .unwrap();
   pool.install(|| {
       // operations here use only 4 threads
   });
   ```

4. **Parallel sort**: The `sort_by` method in Chapter 2 uses sequential sorting. Rayon provides `par_sort_by` -- integrate it and benchmark the improvement on 10M rows.

## Summary

In this chapter, you learned:

- **Why the GIL blocks parallelism**: Python's reference counting requires mutual exclusion; the GIL provides it at the cost of single-threaded execution
- **How ctypes bypasses the GIL**: calling a ctypes function automatically releases the GIL, allowing Rust to use all CPU cores
- **Rayon's work-stealing scheduler**: `par_iter()` distributes work across a thread pool; idle threads steal work from busy threads for load balancing
- **Parallel data structures**: `DashMap` provides a concurrent hash map for parallel groupby operations
- **SIMD auto-vectorization**: contiguous Vec<f64> data enables the compiler to generate instructions that process 4-8 values per clock cycle
- **Combined performance model**: columnar storage + Rayon + SIMD delivers 20-90x speedup over Python for data engineering workloads

The three levels of parallelism (multi-core, SIMD, cache efficiency) are not additive -- they are **multiplicative**. Each level amplifies the others, producing speedups that no single optimization could achieve.

---

**Next**: Chapter 6 will put everything together in a complete ETL pipeline -- loading data from multiple sources, transforming it with the parallel engine, and exporting results via Arrow to downstream systems.
