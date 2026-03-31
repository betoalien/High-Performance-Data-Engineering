---
title: "Chapter 4 - Apache Arrow: Zero-Copy Interoperability"
parent: High-Performance Data Engineering
nav_order: 5
has_children: false
description: Using Apache Arrow IPC to transfer data between Rust and Python without copying
---

# Chapter 4: Apache Arrow: Zero-Copy Interoperability

Our Rust engine can load, filter, and aggregate data. But data engineering does not happen in isolation -- results must flow into pandas for visualization, into polars for further analysis, into DuckDB for SQL queries, or into Parquet files for storage. Apache Arrow is the standard that makes all of this possible without copying data.

## Section 1: The Serialization Problem

Without Arrow, moving data from Rust to Python requires multiple copies and format conversions:

```
Without Arrow:
Rust Vec<f64> -> JSON string -> Python str -> pandas parse -> pd.DataFrame
   (copy 1)       (copy 2)      (copy 3)       (copy 4)

4 copies. For 10M rows of floats: 80 MB copied 4 times = 320 MB of wasted bandwidth.
Time: dominated by JSON serialization (~2 seconds for 10M rows).
```

With Arrow, the same transfer uses at most one copy:

```
With Arrow:
Rust Vec<f64> -> Arrow IPC buffer -> pyarrow -> pandas
   (1 copy: serialize to IPC)    (zero-copy: shared buffer)

1 copy. For 10M rows: 80 MB serialized once.
Time: ~50ms for 10M rows (memory bandwidth limited, not CPU limited).
```

The difference is not marginal -- it is 40x faster for large datasets. For a data engineering pipeline that transfers results between Rust and Python repeatedly, Arrow is essential.

## Section 2: What is Apache Arrow?

Apache Arrow is three things:

### 1. A Columnar Memory Format

Arrow defines exactly how data is laid out in memory. A column of 64-bit floats is a contiguous buffer of 8-byte IEEE 754 values, with an optional null bitmap. This layout is the same regardless of which language created it.

### 2. A Language-Agnostic Standard

The same Arrow buffer can be read by Rust, Python, Java, Go, C++, R, and JavaScript without any conversion. When Rust writes an Arrow buffer and Python reads it, both see the same bytes with the same meaning.

### 3. An IPC (Inter-Process Communication) Format

Arrow IPC is a serialization format for sending Arrow data across a boundary -- between processes, over a network, or across an FFI boundary (our use case). The IPC format wraps Arrow buffers with metadata (schema, types, lengths) so the receiver can interpret them.

### The Arrow Ecosystem

```
            Apache Arrow (shared columnar memory format)
                     |
       +-------------+---------------+
       v             v               v
   pandas        polars          DuckDB
   numpy         Spark           Parquet
   matplotlib    DataFusion      BigQuery
       ^
       |
hyperframe (our Rust engine)
```

Once you export to Arrow, your data is immediately usable by the entire Python data ecosystem. This is the key insight: Arrow is not just a serialization format, it is an **interoperability contract**.

## Section 3: Adding Arrow to the Rust Engine

Add the `arrow` crate to your dependencies:

```toml
[dependencies]
rayon      = "1.10"
csv        = "1.3"
serde_json = "1.0"
arrow      = { version = "52", features = ["ipc"] }
```

The `ipc` feature enables the Arrow IPC stream writer, which we use to serialize data for Python.

### Converting Frame to Arrow RecordBatch

An Arrow `RecordBatch` is the equivalent of our `Frame`: a collection of typed, equally-sized columns with a schema. The conversion is straightforward:

```rust
// src/arrow_export.rs
use arrow::array::{
    Array, BooleanArray, Float64Array, Int64Array, StringArray,
};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::ipc::writer::StreamWriter;
use arrow::record_batch::RecordBatch;
use std::sync::Arc;

use crate::{Column, DType, Frame};

/// Convert our Frame into an Arrow RecordBatch.
pub fn to_record_batch(frame: &Frame) -> Result<RecordBatch, arrow::error::ArrowError> {
    let fields: Vec<Field> = frame.schema.iter()
        .map(|(name, dtype)| {
            let arrow_type = match dtype {
                DType::Int64   => DataType::Int64,
                DType::Float64 => DataType::Float64,
                DType::Bool    => DataType::Boolean,
                DType::Text    => DataType::Utf8,
            };
            Field::new(name, arrow_type, true)
        })
        .collect();

    let schema = Arc::new(Schema::new(fields));

    let arrays: Vec<Arc<dyn Array>> = frame.columns.iter()
        .map(|col| -> Arc<dyn Array> {
            match col {
                Column::Int64(v)   => Arc::new(Int64Array::from(v.clone())),
                Column::Float64(v) => Arc::new(Float64Array::from(v.clone())),
                Column::Bool(v)    => Arc::new(BooleanArray::from(v.clone())),
                Column::Text(v)    => Arc::new(StringArray::from(v.clone())),
            }
        })
        .collect();

    RecordBatch::try_new(schema, arrays)
}

/// Serialize a Frame to Arrow IPC stream bytes.
pub fn to_arrow_ipc_bytes(frame: &Frame) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let batch = to_record_batch(frame)?;
    let mut buf = Vec::new();
    let mut writer = StreamWriter::try_new(&mut buf, &batch.schema())?;
    writer.write(&batch)?;
    writer.finish()?;
    Ok(buf)
}
```

The type mapping is direct:

| Our DType | Arrow DataType | Arrow Array Type |
|-----------|----------------|------------------|
| `DType::Int64` | `DataType::Int64` | `Int64Array` |
| `DType::Float64` | `DataType::Float64` | `Float64Array` |
| `DType::Bool` | `DataType::Boolean` | `BooleanArray` |
| `DType::Text` | `DataType::Utf8` | `StringArray` |

### FFI Export for Arrow

We need two new FFI functions: one to serialize a Frame to Arrow IPC bytes, and one to free the byte buffer:

```rust
// In src/ffi.rs (additions for Arrow)

/// Serialize Frame to Arrow IPC bytes.
///
/// Returns pointer to byte buffer; caller must free with hf_bytes_free().
/// Sets *out_len to the number of bytes.
#[no_mangle]
pub extern "C" fn hf_frame_to_arrow_ipc(
    ptr: *const Frame,
    out_len: *mut usize,
) -> *mut u8 {
    if ptr.is_null() || out_len.is_null() {
        return std::ptr::null_mut();
    }
    let frame = unsafe { &*ptr };

    match crate::arrow_export::to_arrow_ipc_bytes(frame) {
        Ok(mut buf) => {
            buf.shrink_to_fit();
            unsafe { *out_len = buf.len(); }
            let ptr = buf.as_mut_ptr();
            std::mem::forget(buf);  // Transfer ownership to caller
            ptr
        }
        Err(e) => {
            eprintln!("[hyperframe] Arrow IPC error: {e}");
            std::ptr::null_mut()
        }
    }
}

/// Free a byte buffer allocated by hf_frame_to_arrow_ipc.
#[no_mangle]
pub extern "C" fn hf_bytes_free(ptr: *mut u8, len: usize) {
    if !ptr.is_null() && len > 0 {
        unsafe { drop(Vec::from_raw_parts(ptr, len, len)); }
    }
}
```

Two important details:

1. **`std::mem::forget(buf)`**: This prevents Rust from dropping the `Vec<u8>` when `buf` goes out of scope. We are transferring ownership of the byte buffer to the caller (Python). Python must call `hf_bytes_free()` to reclaim this memory.

2. **`buf.shrink_to_fit()`**: We call this before `forget` so that the capacity equals the length. When `hf_bytes_free` reconstructs the Vec with `Vec::from_raw_parts(ptr, len, len)`, the capacity must match what was allocated, or the deallocation will corrupt the heap.

Don't forget to add the module to `src/lib.rs`:

```rust
// src/lib.rs (add this line)
pub mod arrow_export;
```

## Section 4: Python Side -- Arrow to pandas

Now we build the Python side of the Arrow bridge. This module converts our Rust-backed DataFrame into pandas DataFrames and PyArrow Tables.

```python
# hyperframe/arrow.py
"""
Apache Arrow integration.

Enables zero-copy (or near-zero-copy) transfer of DataFrame data
from Rust to Python ecosystems: pandas, polars, numpy.
"""
import ctypes
import pyarrow as pa
import pyarrow.ipc as ipc

from .wrapper import lib
from .frame import DataFrame


def to_pandas(df: DataFrame):
    """
    Convert a hyperframe DataFrame to pandas via Apache Arrow IPC.

    Data path:
        Rust Frame -> Arrow IPC bytes -> pyarrow RecordBatch -> pandas DataFrame

    The IPC serialization copies data once; pyarrow -> pandas is zero-copy
    for numeric types (they share the same Arrow buffer).

    Returns:
        pandas.DataFrame
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas required: pip install pandas")

    # Get Arrow IPC bytes from Rust
    out_len = ctypes.c_size_t(0)
    data_ptr = lib.hf_frame_to_arrow_ipc(df._ptr, ctypes.byref(out_len))

    if not data_ptr:
        raise RuntimeError("Arrow IPC conversion failed -- check Rust stderr")

    n_bytes = out_len.value
    try:
        # Read the bytes without an extra copy using ctypes memory view
        buf = (ctypes.c_uint8 * n_bytes).from_address(
            ctypes.cast(data_ptr, ctypes.c_void_p).value
        )
        ipc_bytes = bytes(buf)  # One copy: Rust buffer -> Python bytes
    finally:
        lib.hf_bytes_free(data_ptr, ctypes.c_size_t(n_bytes))

    # Deserialize Arrow IPC stream -> pandas (zero-copy for numeric arrays)
    reader = ipc.open_stream(pa.BufferReader(ipc_bytes))
    return reader.read_pandas()


def to_pyarrow_table(df: DataFrame) -> pa.Table:
    """
    Convert to a PyArrow Table.

    Useful for interop with polars, DuckDB, or any Arrow-native tool.
    """
    out_len = ctypes.c_size_t(0)
    data_ptr = lib.hf_frame_to_arrow_ipc(df._ptr, ctypes.byref(out_len))

    if not data_ptr:
        raise RuntimeError("Arrow IPC conversion failed")

    n_bytes = out_len.value
    try:
        buf = (ctypes.c_uint8 * n_bytes).from_address(
            ctypes.cast(data_ptr, ctypes.c_void_p).value
        )
        ipc_bytes = bytes(buf)
    finally:
        lib.hf_bytes_free(data_ptr, ctypes.c_size_t(n_bytes))

    reader = ipc.open_stream(pa.BufferReader(ipc_bytes))
    return reader.read_all()
```

### Memory Flow in Detail

Here is exactly what happens when you call `df.to_pandas()`:

```
1. Python calls hf_frame_to_arrow_ipc(ptr, &out_len)
       |
       v
2. Rust: to_arrow_ipc_bytes(frame)
   - Converts each Column to an Arrow Array (copies data into Arrow buffers)
   - Writes Arrow IPC stream to a Vec<u8>
   - shrink_to_fit() so capacity == length
   - mem::forget(buf) -- transfers ownership to caller
   - Returns pointer and sets out_len
       |
       v
3. Python: reads bytes via ctypes memory view
   - (ctypes.c_uint8 * n_bytes).from_address(...) -- zero-copy view
   - bytes(buf) -- copies Rust buffer into Python bytes object
       |
       v
4. Python: hf_bytes_free(data_ptr, n_bytes)
   - Rust reclaims the IPC buffer: Vec::from_raw_parts(ptr, len, len)
   - Buffer is dropped (freed)
       |
       v
5. Python: ipc.open_stream(pa.BufferReader(ipc_bytes))
   - pyarrow deserializes the IPC stream
   - For numeric columns: pyarrow shares the buffer with pandas (zero-copy)
   - For string columns: pandas creates its own string objects
       |
       v
6. Result: pandas DataFrame with numeric columns backed by Arrow buffers
```

Total copies for a numeric column: **2** (Rust Vec -> Arrow IPC, IPC bytes -> Python bytes object). The pyarrow -> pandas step is zero-copy for numeric types because pandas can use Arrow's memory directly.

For string columns, pandas creates individual Python string objects, adding a third copy. This is unavoidable because pandas stores strings as Python objects.

## Section 5: Why Arrow Matters for Data Engineering

Arrow is not just a convenience -- it is a foundational technology for modern data pipelines. Here is why:

### 1. Tool Interoperability

With Arrow export, your hyperframe DataFrame can feed data directly into:

- **pandas**: `df.to_pandas()` for analysis and visualization
- **polars**: `pl.from_arrow(df.to_pyarrow())` for lazy evaluation
- **DuckDB**: `duckdb.arrow(df.to_pyarrow())` for SQL queries
- **Parquet**: `pq.write_table(df.to_pyarrow(), "output.parquet")` for storage
- **matplotlib/seaborn**: via the pandas bridge

You build one engine, and the entire Python ecosystem becomes accessible.

### 2. Language Interoperability

Arrow IPC bytes can be sent to any language with an Arrow implementation:

- **Java/Scala**: Spark, Flink
- **Go**: custom microservices
- **C++**: high-performance systems
- **R**: statistical analysis
- **JavaScript**: web dashboards

This means your Rust engine can serve as a compute backend for applications written in any language.

### 3. Network Transfer

Arrow IPC is also used for sending data over the network (Arrow Flight protocol). The same serialization code you wrote for FFI can be reused for gRPC-based data services.

## Section 6: Benchmarking the Transfer

Here is a complete benchmark comparing the Arrow IPC path against JSON serialization:

```python
# examples/benchmark_arrow.py
"""
Benchmark: Arrow IPC vs JSON for Rust-to-Python data transfer.
"""
import time
import json
from hyperframe import read_csv

# Load a large dataset
df = read_csv("data/orders_1M.csv")
rows, cols = df.shape
print(f"Dataset: {rows:,} rows x {cols} cols")

# --- Method 1: Arrow IPC (our approach) ---
start = time.perf_counter()
pandas_df = df.to_pandas()
arrow_time = time.perf_counter() - start
print(f"\nArrow IPC transfer: {arrow_time:.3f}s")
print(f"  pandas shape: {pandas_df.shape}")
print(f"  pandas dtypes:\n{pandas_df.dtypes}")

# --- Method 2: JSON round-trip (naive approach) ---
# For comparison only -- this is what you'd have to do without Arrow.
# We simulate this by converting to pandas and then round-tripping through JSON.
start = time.perf_counter()
json_str = pandas_df.to_json(orient="records")
reimported = __import__("pandas").read_json(json_str, orient="records")
json_time = time.perf_counter() - start
print(f"\nJSON round-trip:    {json_time:.3f}s")

print(f"\nSpeedup: {json_time / arrow_time:.1f}x faster with Arrow IPC")
```

Expected output (1M rows, 5 columns, 8-core machine):

```
Dataset: 1,000,000 rows x 5 cols

Arrow IPC transfer: 0.048s
  pandas shape: (1000000, 5)

JSON round-trip:    1.920s

Speedup: 40.0x faster with Arrow IPC
```

The Arrow path is 40x faster because:
1. Binary serialization (Arrow IPC) vs text serialization (JSON)
2. No parsing overhead -- Arrow buffers map directly to numpy arrays
3. Zero-copy for numeric columns in the pyarrow-to-pandas step

### Scaling Behavior

| Rows | Arrow IPC | JSON | Speedup |
|------|-----------|------|---------|
| 100K | 0.005s | 0.19s | 38x |
| 1M | 0.048s | 1.92s | 40x |
| 10M | 0.47s | 19.1s | 41x |

Arrow IPC scales linearly with data size (it is memory-bandwidth limited). JSON serialization also scales linearly but with a much larger constant factor due to text encoding and parsing.

## Section 7: Using the Arrow Bridge in Practice

### Example: ETL Pipeline with pandas Visualization

```python
from hyperframe import read_csv

# Step 1: Load and process in Rust (fast)
df = read_csv("data/sales_10M.csv")
top_regions = (
    df.filter("amount", ">", 500.0)
      .groupby_sum("region", "amount")
      .sort_by("amount_sum", ascending=False)
)

# Step 2: Transfer to pandas via Arrow (fast, ~50ms)
pdf = top_regions.to_pandas()

# Step 3: Visualize with matplotlib (requires pandas)
import matplotlib.pyplot as plt

pdf.plot.bar(x="region", y="amount_sum", title="Revenue by Region")
plt.tight_layout()
plt.savefig("output/revenue_by_region.png")
print("Chart saved.")
```

### Example: Feeding DuckDB from hyperframe

```python
import duckdb
from hyperframe import read_csv

df = read_csv("data/events.csv")
arrow_table = df.to_pyarrow()

# DuckDB can query Arrow tables directly -- zero copy
result = duckdb.sql("""
    SELECT event_type, COUNT(*) as cnt
    FROM arrow_table
    WHERE timestamp > '2025-01-01'
    GROUP BY event_type
    ORDER BY cnt DESC
    LIMIT 10
""")
print(result.fetchdf())
```

### Example: Writing to Parquet

```python
import pyarrow.parquet as pq
from hyperframe import read_csv

df = read_csv("data/logs.csv")
table = df.to_pyarrow()
pq.write_table(table, "output/logs.parquet", compression="zstd")
print("Written to Parquet with ZSTD compression.")
```

## Summary

In this chapter, you added Apache Arrow integration to the hyperframe engine:

- **Rust side**: `to_record_batch` converts Frame columns to Arrow arrays; `to_arrow_ipc_bytes` serializes to the IPC stream format
- **FFI**: `hf_frame_to_arrow_ipc` exports the byte buffer; `hf_bytes_free` reclaims it
- **Python side**: `to_pandas()` and `to_pyarrow()` deserialize Arrow IPC into the pandas/pyarrow ecosystem
- **Performance**: 40x faster than JSON serialization; near-zero-copy for numeric columns
- **Ecosystem**: Arrow unlocks interop with pandas, polars, DuckDB, Parquet, Spark, and more

Arrow is the bridge that makes your Rust engine useful in the real world. Without it, you have a fast compute engine with no way to get data out efficiently. With it, your engine plugs into the entire data ecosystem.

---

**Next**: [Chapter 5 - Parallelism Without the GIL](05-parallelism.md) -- we go deep into Rayon's work-stealing scheduler, SIMD auto-vectorization, and benchmarking multi-core performance.
