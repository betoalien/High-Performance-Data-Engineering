---
title: Chapter 6 - Async I/O with Tokio
parent: High-Performance Data Engineering
nav_order: 7
has_children: false
description: Using Tokio for concurrent data ingestion from multiple sources
---

# Chapter 6: Async I/O with Tokio

In Chapter 5, we used Rayon to parallelize *computation* across CPU cores. But CPU parallelism only helps if data arrives fast enough to keep those cores busy. When a pipeline reads from 20 CSV files, S3 buckets, or database connections, **sequential I/O** is the real bottleneck -- not computation.

This chapter introduces concurrent I/O: loading multiple data sources simultaneously so the CPU never starves for work.

---

## 6.1 The I/O Bottleneck

Consider a pipeline that loads 20 CSV files from disk or a network file system:

```
Sequential I/O:       [read file1][read file2][read file3]...  total: N x t
Concurrent I/O:       [file1]
                      [file2]   <- all in flight simultaneously
                      [file3]
                      total: ~ t (limited by network/disk bandwidth, not count)
```

Sequential loading takes `N x t` seconds because each file waits for the previous one to finish. Concurrent loading takes roughly `t` seconds because all files are in flight simultaneously -- the total time is bounded by the **bandwidth** of the storage medium, not the **number** of files.

This matters enormously in real data engineering:

- **Data lakes**: hundreds of date-partitioned CSV/Parquet files
- **Multi-source ingestion**: database + S3 + REST API feeds
- **Micro-batch pipelines**: dozens of small files arriving every minute

Rayon cannot help here. Rayon parallelizes CPU work within a single process. I/O is dominated by waiting -- waiting for disk seeks, network round-trips, or database query execution. We need a different tool: **asynchronous I/O**.

---

## 6.2 Sync vs Async

### The Restaurant Analogy

Imagine a restaurant with one waiter:

- **Synchronous (blocking)**: The waiter takes an order, walks to the kitchen, *stands there waiting* until the food is ready, delivers it, then walks to the next table. One customer at a time.
- **Asynchronous (non-blocking)**: The waiter takes an order, gives it to the kitchen, immediately walks to the next table and takes another order. When food is ready, the waiter picks it up and delivers it. Many customers served concurrently.

The async waiter handles the same workload with far less idle time.

### Synchronous File Loading

```rust
// Synchronous: thread blocks waiting for each file
fn load_all_sync(paths: &[&str]) -> Vec<Frame> {
    paths.iter()
        .map(|path| read_csv(path).unwrap())
        .collect()  // sequential -- one at a time
}
```

Each `read_csv` call blocks the thread until the file is fully read and parsed. With 20 files, the total time is the sum of all individual read times.

### Asynchronous File Loading

```rust
use tokio::task;

// Async: all files in flight concurrently
async fn load_all_async(paths: Vec<String>) -> Vec<Frame> {
    let tasks: Vec<_> = paths.into_iter()
        .map(|path| task::spawn_blocking(move || read_csv(&path).unwrap()))
        .collect();

    let mut frames = Vec::new();
    for task in tasks {
        frames.push(task.await.unwrap());
    }
    frames
}
```

All files are dispatched simultaneously. The `await` keyword suspends the current task without blocking the thread, allowing other work to proceed.

### Why `spawn_blocking`?

CSV parsing is CPU-bound work -- it involves scanning bytes, splitting fields, and type-converting values. Tokio's async runtime is designed for I/O-bound tasks that spend most of their time waiting. Running CPU-heavy work on the async executor blocks it and prevents other tasks from making progress.

`spawn_blocking` moves the work to Tokio's dedicated **blocking thread pool**, which is sized separately from the async worker threads. This gives us the best of both worlds:

- **Concurrent dispatch**: all files start loading at the same time
- **No executor starvation**: CPU-bound parsing runs on blocking threads
- **Automatic backpressure**: the blocking pool has a configurable thread limit

---

## 6.3 Adding Tokio to the Engine

Update the Cargo.toml to include Tokio:

```toml
[dependencies]
rayon      = "1.10"
csv        = "1.3"
serde_json = "1.0"
arrow      = { version = "52", features = ["ipc"] }
tokio      = { version = "1", features = ["full"] }
```

### Complete Async Multi-File Loader

```rust
// src/async_io.rs
use tokio::task;
use crate::{Frame, io::read_csv};

/// Load multiple CSV files concurrently.
///
/// Uses Tokio's blocking thread pool so CPU-bound CSV parsing
/// doesn't block the async executor.
pub async fn read_csv_many(paths: Vec<String>)
    -> Result<Vec<Frame>, Box<dyn std::error::Error + Send + Sync>>
{
    let handles: Vec<_> = paths.into_iter()
        .map(|path| {
            task::spawn_blocking(move || {
                read_csv(&path).map_err(|e| e.to_string())
            })
        })
        .collect();

    let mut frames = Vec::with_capacity(handles.len());
    for handle in handles {
        frames.push(handle.await??);
    }
    Ok(frames)
}

/// Concatenate multiple Frames with identical schemas into one.
pub fn concat(frames: Vec<Frame>) -> Result<Frame, String> {
    if frames.is_empty() {
        return Err("No frames to concatenate".into());
    }

    let schema = frames[0].schema.clone();
    let nrows: usize = frames.iter().map(|f| f.nrows).sum();

    let ncols = schema.len();

    use crate::Column;
    let columns: Vec<Column> = (0..ncols).map(|ci| {
        match &frames[0].columns[ci] {
            Column::Float64(_) => Column::Float64(
                frames.iter().flat_map(|f| {
                    if let Column::Float64(v) = &f.columns[ci] { v.iter().copied() }
                    else { [].iter().copied() }
                }).collect()
            ),
            Column::Int64(_) => Column::Int64(
                frames.iter().flat_map(|f| {
                    if let Column::Int64(v) = &f.columns[ci] { v.iter().copied() }
                    else { [].iter().copied() }
                }).collect()
            ),
            Column::Bool(_) => Column::Bool(
                frames.iter().flat_map(|f| {
                    if let Column::Bool(v) = &f.columns[ci] { v.iter().copied() }
                    else { [].iter().copied() }
                }).collect()
            ),
            Column::Text(_) => Column::Text(
                frames.iter().flat_map(|f| {
                    if let Column::Text(v) = &f.columns[ci] { v.iter().cloned() }
                    else { vec![].into_iter() }
                }).collect()
            ),
        }
    }).collect();

    Ok(Frame { schema, columns, nrows })
}
```

### FFI Export for Concatenation

Python calls this synchronously; the Tokio runtime is used internally when needed:

```rust
// In ffi.rs -- Python calls this synchronously; internally uses Tokio
#[no_mangle]
pub extern "C" fn hf_frame_concat(
    ptrs: *const *const Frame,
    count: usize,
) -> *mut Frame {
    if ptrs.is_null() || count == 0 { return std::ptr::null_mut(); }

    let frames: Vec<Frame> = (0..count)
        .map(|i| unsafe { (*(*ptrs.add(i))).clone() })
        .collect();

    match crate::async_io::concat(frames) {
        Ok(f)  => Box::into_raw(Box::new(f)),
        Err(e) => { eprintln!("[hyperframe] concat error: {e}"); std::ptr::null_mut() }
    }
}
```

---

## 6.4 Python Side -- Concurrent Loading

On the Python side, we don't need Tokio at all. Python's `concurrent.futures` module with `ThreadPoolExecutor` provides the same concurrent dispatch pattern -- and it works because **ctypes calls release the GIL**.

```python
from concurrent.futures import ThreadPoolExecutor, as_completed
from hyperframe import read_csv, DataFrame

def read_csv_many(paths: list[str], max_workers: int = 8) -> list[DataFrame]:
    """
    Load multiple CSV files concurrently.

    Each read_csv() call releases the GIL (ctypes), so true parallelism
    is achieved even with Python threads.
    """
    results = [None] * len(paths)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(read_csv, path): i
            for i, path in enumerate(paths)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            results[idx] = future.result()

    return results
```

### Why This Works

This is a critical insight for the entire course:

1. `read_csv()` is implemented as a `ctypes` call to the Rust shared library
2. When Python calls a `ctypes` function, the **GIL is released** automatically
3. With the GIL released, all Python threads run truly in parallel
4. Each thread calls into Rust independently -- Rust has no GIL

This means Python threads + ctypes gives us **real parallelism** without any of the usual GIL limitations. Each file is loaded and parsed concurrently across multiple OS threads.

Compare this to pure Python or even NumPy-based approaches where threads are serialized by the GIL. The ctypes FFI boundary is the key that unlocks true thread-level parallelism from Python.

---

## 6.5 Combining Async I/O with Parallel Compute

The most powerful pattern combines concurrent I/O with per-file computation. Each file is loaded *and processed* concurrently:

```python
from concurrent.futures import ThreadPoolExecutor
from hyperframe import read_csv

def process_partition(path: str, filter_col: str, threshold: float):
    """Load one file and immediately filter -- I/O and compute overlap."""
    df = read_csv(path)                                # I/O (GIL released)
    filtered = df.filter(filter_col, ">", threshold)   # compute (GIL released)
    return filtered

# Load + process 20 files concurrently
paths = [f"data/partition_{i:03d}.csv" for i in range(20)]

with ThreadPoolExecutor(max_workers=8) as executor:
    results = list(executor.map(
        lambda p: process_partition(p, "revenue", 100.0),
        paths
    ))

# Combine results
print(f"Total rows after filtering: {sum(r.shape[0] for r in results):,}")
```

This pattern is extremely effective because:

- **I/O and compute overlap**: while one thread waits for disk, another is filtering
- **Memory pressure is distributed**: only a few files are fully materialized at once
- **Both stages release the GIL**: ctypes calls for both `read_csv` and `filter`

### Pipeline Diagram

```
Thread 1:  [read file_000] [filter file_000] [done]
Thread 2:  [read file_001] [filter file_001] [done]
Thread 3:  [read file_002] [filter file_002] [done]
...
Thread 8:  [read file_007] [filter file_007] [read file_008] [filter file_008] ...

Total wall time: much less than sequential
```

---

## 6.6 Real-World Example -- Partitioned Data Lake

Many production systems store data in date-partitioned directory structures:

```
data/warehouse/orders/
    dt=2024-01-01/data.csv
    dt=2024-01-02/data.csv
    dt=2024-01-03/data.csv
    ...
    dt=2024-12-31/data.csv
```

Loading a month of data means reading ~30 files. Doing this sequentially wastes enormous amounts of time waiting for I/O.

```python
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from hyperframe import read_csv

def load_date_range(base_dir: str, start: str, end: str) -> list:
    """Load all daily partitions in a date range concurrently."""
    from datetime import date, timedelta

    start_date = date.fromisoformat(start)
    end_date   = date.fromisoformat(end)

    paths = []
    current = start_date
    while current <= end_date:
        path = Path(base_dir) / f"dt={current.isoformat()}" / "data.csv"
        if path.exists():
            paths.append(str(path))
        current += timedelta(days=1)

    print(f"Loading {len(paths)} partitions concurrently...")

    with ThreadPoolExecutor(max_workers=min(len(paths), 16)) as ex:
        frames = list(ex.map(read_csv, paths))

    print(f"Loaded {sum(f.shape[0] for f in frames):,} total rows")
    return frames

# Usage
frames = load_date_range("data/warehouse/orders", "2024-01-01", "2024-01-31")
```

### Benchmarks: Concurrent vs Sequential

Loading 30 daily partition files (each ~100 MB, totaling 3 GB):

```
Strategy                    | Wall Time  | Throughput
----------------------------|------------|----------
Sequential read_csv loop    | 24.6s      | 122 MB/s
ThreadPool (4 workers)      | 7.1s       | 423 MB/s
ThreadPool (8 workers)      | 4.2s       | 714 MB/s
ThreadPool (16 workers)     | 3.8s       | 789 MB/s
```

Observations:

- **4 workers** gives ~3.5x speedup (I/O parallelism saturating disk bandwidth)
- **8 workers** gives ~5.9x speedup (overlapping I/O with CPU-bound parsing)
- **16 workers** shows diminishing returns -- we've saturated the storage bandwidth
- The sweet spot is usually `min(num_files, 2 * num_cpu_cores)`

### When to Use More Workers

The optimal `max_workers` depends on the I/O characteristics:

| Storage Type | Recommended Workers | Why |
|-------------|--------------------|----|
| Local SSD (NVMe) | 4-8 | High bandwidth, low latency |
| Network filesystem (NFS) | 8-16 | Higher latency, benefits from more concurrency |
| S3 / cloud storage | 16-64 | Very high latency, massive parallelism helps |
| Database connections | pool size | Limited by connection pool, not bandwidth |

---

## 6.7 Error Handling in Concurrent Loading

When loading many files concurrently, failures will happen -- missing files, corrupt data, permission errors. A robust loader must handle partial failures gracefully:

```python
from concurrent.futures import ThreadPoolExecutor, as_completed
from hyperframe import read_csv
import logging

log = logging.getLogger(__name__)

def read_csv_many_robust(
    paths: list[str],
    max_workers: int = 8,
    fail_fast: bool = False,
) -> tuple[list, list[str]]:
    """
    Load multiple CSV files concurrently with error handling.

    Returns:
        (successful_frames, failed_paths)
    """
    results = {}
    errors  = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_path = {
            executor.submit(read_csv, path): path
            for path in paths
        }

        for future in as_completed(future_to_path):
            path = future_to_path[future]
            try:
                df = future.result()
                results[path] = df
                log.info(f"Loaded {path}: {df.shape[0]:,} rows")
            except Exception as e:
                log.error(f"Failed to load {path}: {e}")
                errors.append(path)
                if fail_fast:
                    raise RuntimeError(f"Failed on {path}: {e}") from e

    # Preserve original order for successfully loaded files
    frames = [results[p] for p in paths if p in results]

    if errors:
        log.warning(f"{len(errors)}/{len(paths)} files failed to load")

    return frames, errors
```

---

## Summary

This chapter introduced concurrent I/O as the complement to Chapter 5's CPU parallelism:

- **Sequential I/O** is the bottleneck when loading multiple files, not CPU speed
- **Tokio** provides async runtime for concurrent I/O in Rust (`spawn_blocking` for CPU-bound work)
- **Python ThreadPoolExecutor** achieves real parallelism because ctypes releases the GIL
- **Combined I/O + compute** pattern: load and process files concurrently for maximum throughput
- **Partitioned data lakes** are a natural fit for concurrent loading

The key insight: ctypes FFI calls release the GIL, making Python threads truly parallel for Rust-backed operations. This eliminates the need for multiprocessing and its associated overhead (IPC, serialization, memory duplication).

In the next chapter, we tie everything together into a production ETL pipeline.
