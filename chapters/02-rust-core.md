---
title: Chapter 2 - Columnar Storage from Scratch in Rust
parent: High-Performance Data Engineering
nav_order: 3
has_children: false
description: Building a type-safe columnar data engine in Rust with parallel compute kernels
---

# Chapter 2: Columnar Storage from Scratch in Rust

This chapter builds the complete Rust engine. Every code block is complete and compilable. By the end, you will have a working `hyperframe-core` crate that reads CSV files, filters rows in parallel, computes aggregations, and exports 12 FFI functions for Python to call.

## Section 1: Why Columnar Storage

The single most important architectural decision in a data engine is how data is laid out in memory. There are two fundamental approaches:

### Row-Oriented Storage

This is what you get with a Python list of dicts, a SQL database row page, or a JSON array:

```
Row-oriented (bad for analytics):
| id:1 | name:"Alice" | price:42.0 | qty:3 |
| id:2 | name:"Bob"   | price:15.0 | qty:1 |
| id:3 | name:"Carol" | price:99.0 | qty:7 |
  ^ To sum prices, CPU loads ALL columns
```

To compute `SUM(price)`, the CPU must load every field of every row into cache, even though it only needs the price column. Most of the data in each cache line is irrelevant.

### Columnar Storage

This is what Arrow, Parquet, and our engine use:

```
Columnar (good for analytics):
prices: | 42.0 | 15.0 | 99.0 | ... |
  ^ To sum prices, CPU loads ONLY this contiguous array
  ^ CPU can use SIMD: process 4 doubles per instruction
```

All price values are adjacent in memory. A single 64-byte cache line holds 8 doubles. The CPU prefetcher can predict the access pattern and load the next cache line before it is needed. SIMD instructions can process 4 (AVX) or 8 (AVX-512) values per clock cycle.

For analytical workloads (filtering, aggregation, scanning), columnar storage is dramatically faster.

## Section 2: The Type System

Every column has a type. We define a simple enum that covers the most common data engineering types:

```rust
// src/dtype.rs

/// Supported data types in our engine.
/// Kept intentionally simple -- add Decimal, Date etc. as exercises.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    Int64,
    Float64,
    Bool,
    Text,
}

impl std::fmt::Display for DType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DType::Int64   => write!(f, "Int64"),
            DType::Float64 => write!(f, "Float64"),
            DType::Bool    => write!(f, "Bool"),
            DType::Text    => write!(f, "Text"),
        }
    }
}
```

This enum serves two purposes:
1. **Schema definition**: describing what type each column holds
2. **FFI serialization**: the `Display` impl lets us serialize schemas to JSON for Python

**Exercise**: Add `Date` (days since epoch as `i64`) and `Timestamp` (microseconds since epoch as `i64`) variants. What changes are needed in `Column` to support them?

## Section 3: Column Storage

Each column variant owns a contiguous `Vec<T>`. This is the core of our cache-friendly design:

```rust
// src/column.rs
use crate::DType;

/// A typed column -- one Vec<T> per type.
/// Each variant owns its data contiguously in memory.
#[derive(Debug, Clone)]
pub enum Column {
    Int64(Vec<i64>),
    Float64(Vec<f64>),
    Bool(Vec<bool>),
    Text(Vec<String>),
}

impl Column {
    pub fn len(&self) -> usize {
        match self {
            Column::Int64(v)   => v.len(),
            Column::Float64(v) => v.len(),
            Column::Bool(v)    => v.len(),
            Column::Text(v)    => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn dtype(&self) -> DType {
        match self {
            Column::Int64(_)   => DType::Int64,
            Column::Float64(_) => DType::Float64,
            Column::Bool(_)    => DType::Bool,
            Column::Text(_)    => DType::Text,
        }
    }

    /// Apply a boolean mask, keeping only rows where mask[i] == true
    pub fn filter(&self, mask: &[bool]) -> Column {
        match self {
            Column::Int64(v) => Column::Int64(
                v.iter().zip(mask).filter(|(_, &m)| m).map(|(x, _)| *x).collect()
            ),
            Column::Float64(v) => Column::Float64(
                v.iter().zip(mask).filter(|(_, &m)| m).map(|(x, _)| *x).collect()
            ),
            Column::Bool(v) => Column::Bool(
                v.iter().zip(mask).filter(|(_, &m)| m).map(|(x, _)| *x).collect()
            ),
            Column::Text(v) => Column::Text(
                v.iter().zip(mask).filter(|(_, &m)| m).map(|(x, _)| x.clone()).collect()
            ),
        }
    }
}
```

Key design points:
- `Column::Float64(Vec<f64>)` stores all values contiguously -- perfect for SIMD and cache locality
- `Column::Text(Vec<String>)` stores heap-allocated strings; the `Vec` itself is contiguous (pointers are adjacent), but the string data is on the heap
- The `filter` method creates a **new** column rather than mutating in place, following Rust's preference for immutable data transformations

## Section 4: The Frame

The `Frame` struct is our main data structure -- analogous to a pandas DataFrame or an Arrow RecordBatch:

```rust
// src/frame.rs
use crate::{Column, DType};

/// A columnar table -- the main data structure.
///
/// Invariant: all columns have exactly `nrows` elements.
#[derive(Debug, Clone)]
pub struct Frame {
    /// Column names and types, in order.
    pub schema: Vec<(String, DType)>,
    /// Column data, parallel to schema.
    pub columns: Vec<Column>,
    /// Number of rows.
    pub nrows: usize,
}

impl Frame {
    /// Create an empty Frame with a given schema (0 rows).
    pub fn empty(schema: Vec<(String, DType)>) -> Self {
        let columns = schema.iter().map(|(_, dtype)| match dtype {
            DType::Int64   => Column::Int64(Vec::new()),
            DType::Float64 => Column::Float64(Vec::new()),
            DType::Bool    => Column::Bool(Vec::new()),
            DType::Text    => Column::Text(Vec::new()),
        }).collect();
        Frame { schema, columns, nrows: 0 }
    }

    /// Find column index by name, or None.
    pub fn col_index(&self, name: &str) -> Option<usize> {
        self.schema.iter().position(|(n, _)| n == name)
    }

    /// Apply a boolean mask across all columns.
    pub fn apply_mask(&self, mask: &[bool]) -> Frame {
        assert_eq!(mask.len(), self.nrows, "Mask length must match row count");
        let nrows = mask.iter().filter(|&&b| b).count();
        Frame {
            schema: self.schema.clone(),
            columns: self.columns.iter().map(|c| c.filter(mask)).collect(),
            nrows,
        }
    }
}
```

The `schema` and `columns` vectors are kept in parallel: `schema[i]` describes `columns[i]`. The `nrows` field is redundant (it equals `columns[0].len()`), but keeping it avoids repeated `match` lookups in hot paths.

## Section 5: Parallel Compute with Rayon

This is where the GIL bypass happens. When Python calls a ctypes function, **the GIL is released**. Rust code runs freely on all CPU cores via Rayon's work-stealing thread pool.

```rust
// src/compute.rs
use rayon::prelude::*;
use crate::{Column, DType, Frame};

impl Frame {
    /// Sum a numeric column in parallel.
    ///
    /// Rayon's par_iter() distributes work across all available CPU cores.
    /// This is impossible from Python (GIL blocks multi-core CPU use).
    pub fn sum(&self, col_name: &str) -> Option<f64> {
        let idx = self.col_index(col_name)?;
        match &self.columns[idx] {
            Column::Float64(v) => Some(v.par_iter().sum()),
            Column::Int64(v)   => Some(v.par_iter().map(|&x| x as f64).sum()),
            _ => None,
        }
    }

    /// Compute mean of a numeric column.
    pub fn mean(&self, col_name: &str) -> Option<f64> {
        let idx = self.col_index(col_name)?;
        if self.nrows == 0 { return Some(0.0); }
        match &self.columns[idx] {
            Column::Float64(v) => Some(v.par_iter().sum::<f64>() / v.len() as f64),
            Column::Int64(v)   => Some(v.par_iter().map(|&x| x as f64).sum::<f64>() / v.len() as f64),
            _ => None,
        }
    }

    /// Filter rows where float/int column > threshold.
    /// Builds the boolean mask in parallel, then applies it.
    pub fn filter_gt(&self, col_name: &str, threshold: f64) -> Frame {
        let idx = match self.col_index(col_name) {
            Some(i) => i,
            None => return self.apply_mask(&vec![false; self.nrows]),
        };

        let mask: Vec<bool> = match &self.columns[idx] {
            Column::Float64(v) => v.par_iter().map(|&x| x > threshold).collect(),
            Column::Int64(v)   => v.par_iter().map(|&x| (x as f64) > threshold).collect(),
            _ => vec![false; self.nrows],
        };

        self.apply_mask(&mask)
    }

    /// Filter rows where text column exactly equals value.
    pub fn filter_eq_str(&self, col_name: &str, value: &str) -> Frame {
        let idx = match self.col_index(col_name) {
            Some(i) => i,
            None => return self.apply_mask(&vec![false; self.nrows]),
        };

        let mask: Vec<bool> = match &self.columns[idx] {
            Column::Text(v) => v.par_iter().map(|s| s.as_str() == value).collect(),
            _ => vec![false; self.nrows],
        };

        self.apply_mask(&mask)
    }

    /// Group by a text column and sum a numeric column.
    pub fn groupby_sum(&self, group_col: &str, agg_col: &str)
        -> Result<Frame, String>
    {
        use std::collections::BTreeMap; // BTreeMap gives sorted output for free

        let gi = self.col_index(group_col)
            .ok_or_else(|| format!("Column '{}' not found", group_col))?;
        let ai = self.col_index(agg_col)
            .ok_or_else(|| format!("Column '{}' not found", agg_col))?;

        let groups = match &self.columns[gi] {
            Column::Text(v) => v,
            _ => return Err(format!("'{}' must be a Text column", group_col)),
        };

        let mut sums: BTreeMap<String, f64> = BTreeMap::new();

        match &self.columns[ai] {
            Column::Float64(v) => {
                for (k, &val) in groups.iter().zip(v.iter()) {
                    *sums.entry(k.clone()).or_insert(0.0) += val;
                }
            }
            Column::Int64(v) => {
                for (k, &val) in groups.iter().zip(v.iter()) {
                    *sums.entry(k.clone()).or_insert(0.0) += val as f64;
                }
            }
            _ => return Err(format!("'{}' must be a numeric column", agg_col)),
        }

        let keys: Vec<String> = sums.keys().cloned().collect();
        let totals: Vec<f64> = sums.values().copied().collect();
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

    /// Sort by a column (ascending or descending).
    pub fn sort_by(&self, col_name: &str, ascending: bool) -> Result<Frame, String> {
        let idx = self.col_index(col_name)
            .ok_or_else(|| format!("Column '{}' not found", col_name))?;

        // Build sort indices
        let mut indices: Vec<usize> = (0..self.nrows).collect();

        match &self.columns[idx] {
            Column::Float64(v) => indices.sort_by(|&a, &b| {
                let ord = v[a].partial_cmp(&v[b]).unwrap_or(std::cmp::Ordering::Equal);
                if ascending { ord } else { ord.reverse() }
            }),
            Column::Int64(v) => indices.sort_by(|&a, &b| {
                let ord = v[a].cmp(&v[b]);
                if ascending { ord } else { ord.reverse() }
            }),
            Column::Text(v) => indices.sort_by(|&a, &b| {
                let ord = v[a].cmp(&v[b]);
                if ascending { ord } else { ord.reverse() }
            }),
            Column::Bool(v) => indices.sort_by(|&a, &b| {
                let ord = v[a].cmp(&v[b]);
                if ascending { ord } else { ord.reverse() }
            }),
        }

        // Reorder all columns by the sort indices
        let columns: Vec<Column> = self.columns.iter().map(|col| {
            match col {
                Column::Float64(v) => Column::Float64(indices.iter().map(|&i| v[i]).collect()),
                Column::Int64(v)   => Column::Int64(indices.iter().map(|&i| v[i]).collect()),
                Column::Bool(v)    => Column::Bool(indices.iter().map(|&i| v[i]).collect()),
                Column::Text(v)    => Column::Text(indices.iter().map(|&i| v[i].clone()).collect()),
            }
        }).collect();

        Ok(Frame { schema: self.schema.clone(), columns, nrows: self.nrows })
    }
}
```

Notice that changing `iter()` to `par_iter()` is the only difference between serial and parallel execution. Rayon handles thread creation, work distribution, load balancing, and result merging automatically. Chapter 5 goes deeper into how this works.

## Section 6: CSV Reading

A data engine is useless without I/O. Here is a complete CSV reader that auto-detects column types by sampling the first 200 rows:

```rust
// src/io.rs
use csv::ReaderBuilder;
use crate::{Column, DType, Frame};

/// Load a CSV file into a Frame.
/// Auto-detects column types by sampling the first 200 rows.
pub fn read_csv(path: &str) -> Result<Frame, Box<dyn std::error::Error + Send + Sync>> {
    let mut rdr = ReaderBuilder::new()
        .has_headers(true)
        .trim(csv::Trim::All)
        .from_path(path)?;

    let headers: Vec<String> = rdr.headers()?.iter()
        .map(|s| s.to_string())
        .collect();
    let ncols = headers.len();

    // Collect all records
    let rows: Vec<csv::StringRecord> = rdr.into_records()
        .filter_map(|r| r.ok())
        .collect();
    let nrows = rows.len();

    // Detect types from first 200 rows
    let dtypes: Vec<DType> = (0..ncols)
        .map(|i| detect_dtype(&rows, i))
        .collect();

    // Build typed columns
    let schema: Vec<(String, DType)> = headers.into_iter()
        .zip(dtypes.iter().copied())
        .collect();

    let columns: Vec<Column> = dtypes.iter().enumerate().map(|(i, &dtype)| {
        match dtype {
            DType::Int64 => Column::Int64(
                rows.iter()
                    .map(|r| r.get(i).and_then(|s| s.parse().ok()).unwrap_or(0))
                    .collect()
            ),
            DType::Float64 => Column::Float64(
                rows.iter()
                    .map(|r| r.get(i).and_then(|s| s.parse().ok()).unwrap_or(0.0))
                    .collect()
            ),
            DType::Bool => Column::Bool(
                rows.iter()
                    .map(|r| matches!(
                        r.get(i).map(|s| s.to_ascii_lowercase()).as_deref(),
                        Some("true") | Some("1") | Some("yes")
                    ))
                    .collect()
            ),
            DType::Text => Column::Text(
                rows.iter()
                    .map(|r| r.get(i).unwrap_or("").to_string())
                    .collect()
            ),
        }
    }).collect();

    Ok(Frame { schema, columns, nrows })
}

fn detect_dtype(rows: &[csv::StringRecord], col: usize) -> DType {
    let sample: Vec<&str> = rows.iter()
        .take(200)
        .filter_map(|r| r.get(col))
        .filter(|s| !s.trim().is_empty())
        .collect();

    if sample.is_empty() { return DType::Text; }

    if sample.iter().all(|s| s.parse::<i64>().is_ok()) {
        return DType::Int64;
    }
    if sample.iter().all(|s| s.parse::<f64>().is_ok()) {
        return DType::Float64;
    }
    if sample.iter().all(|s| matches!(
        s.to_ascii_lowercase().as_str(),
        "true" | "false" | "1" | "0" | "yes" | "no"
    )) {
        return DType::Bool;
    }
    DType::Text
}
```

The type detection strategy is deliberately simple: sample the first 200 rows, try parsing as `i64`, then `f64`, then `bool`, and fall back to `Text`. Production engines use more sophisticated heuristics, but this covers the 90% case.

### NDJSON Reader

For the Python `DataFrame(data=[...])` constructor, we also need an NDJSON (newline-delimited JSON) reader:

```rust
// src/io.rs (continued)
use serde_json::Value;

/// Load NDJSON bytes into a Frame.
/// Each line is a JSON object; keys become column names.
pub fn read_ndjson(bytes: &[u8]) -> Result<Frame, Box<dyn std::error::Error + Send + Sync>> {
    let text = std::str::from_utf8(bytes)?;
    let records: Vec<Value> = text.lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| serde_json::from_str(line))
        .collect::<Result<Vec<_>, _>>()?;

    if records.is_empty() {
        return Err("No records found".into());
    }

    // Extract column names from the first record
    let first = records[0].as_object()
        .ok_or("Each line must be a JSON object")?;
    let col_names: Vec<String> = first.keys().cloned().collect();

    // Detect types from values
    let mut schema = Vec::new();
    let mut columns = Vec::new();

    for name in &col_names {
        // Collect all values for this column
        let vals: Vec<&Value> = records.iter()
            .map(|r| r.get(name).unwrap_or(&Value::Null))
            .collect();

        // Detect type from first non-null value
        let dtype = vals.iter().find(|v| !v.is_null()).map_or(DType::Text, |v| {
            if v.is_i64() { DType::Int64 }
            else if v.is_f64() || v.is_number() { DType::Float64 }
            else if v.is_boolean() { DType::Bool }
            else { DType::Text }
        });

        let col = match dtype {
            DType::Int64 => Column::Int64(
                vals.iter().map(|v| v.as_i64().unwrap_or(0)).collect()
            ),
            DType::Float64 => Column::Float64(
                vals.iter().map(|v| v.as_f64().unwrap_or(0.0)).collect()
            ),
            DType::Bool => Column::Bool(
                vals.iter().map(|v| v.as_bool().unwrap_or(false)).collect()
            ),
            DType::Text => Column::Text(
                vals.iter().map(|v| {
                    v.as_str().map(|s| s.to_string())
                        .unwrap_or_else(|| v.to_string())
                }).collect()
            ),
        };

        schema.push((name.clone(), dtype));
        columns.push(col);
    }

    let nrows = records.len();
    Ok(Frame { schema, columns, nrows })
}
```

## Section 7: FFI Exports

These are the 12 functions that Python will call. Every function uses the `#[no_mangle]` attribute and `extern "C"` calling convention to produce C-compatible symbols.

Three critical FFI patterns are used throughout:

1. **`Box::into_raw`**: Transfers ownership from Rust to the caller. Rust will NOT free this memory -- the caller (Python) must call `hf_frame_free` when done.

2. **`Box::from_raw`**: Reclaims ownership for deallocation. Used in `hf_frame_free` to re-create the `Box` and let Rust's drop logic run.

3. **Thread-local string buffer**: When returning strings across the FFI boundary, we store the `CString` in a thread-local `RefCell`. This keeps the string alive until the next FFI call on the same thread, preventing use-after-free without requiring Python to manage the memory.

```rust
// src/ffi.rs
use std::ffi::{CStr, CString, c_char};
use crate::Frame;

/// Thread-local buffer for returning strings across the FFI boundary.
/// This keeps the string alive until the next FFI call on this thread,
/// preventing use-after-free without requiring Python to manage the memory.
thread_local! {
    static STR_BUF: std::cell::RefCell<CString> =
        std::cell::RefCell::new(CString::new("").unwrap());
}

fn return_str(s: String) -> *const c_char {
    STR_BUF.with(|buf| {
        let cs = CString::new(s).unwrap_or_default();
        let ptr = cs.as_ptr();
        *buf.borrow_mut() = cs;
        ptr
    })
}

fn cstr(ptr: *const c_char) -> &'static str {
    if ptr.is_null() { return ""; }
    unsafe { CStr::from_ptr(ptr).to_str().unwrap_or("") }
}

// ---------------------------------------------------------------------------
// Memory management
// ---------------------------------------------------------------------------

/// Release a Frame. Must be called when Python is done with it.
#[no_mangle]
pub extern "C" fn hf_frame_free(ptr: *mut Frame) {
    if !ptr.is_null() {
        unsafe { drop(Box::from_raw(ptr)); }
    }
}

// ---------------------------------------------------------------------------
// Data loading
// ---------------------------------------------------------------------------

/// Load CSV -> Frame. Returns null on error (check stderr for details).
#[no_mangle]
pub extern "C" fn hf_frame_from_csv(path: *const c_char) -> *mut Frame {
    match crate::io::read_csv(cstr(path)) {
        Ok(f)  => Box::into_raw(Box::new(f)),
        Err(e) => { eprintln!("[hyperframe] read_csv error: {e}"); std::ptr::null_mut() }
    }
}

/// Load NDJSON bytes -> Frame.
#[no_mangle]
pub extern "C" fn hf_frame_from_records(
    data: *const u8,
    len: usize,
) -> *mut Frame {
    let bytes = unsafe { std::slice::from_raw_parts(data, len) };
    match crate::io::read_ndjson(bytes) {
        Ok(f)  => Box::into_raw(Box::new(f)),
        Err(e) => { eprintln!("[hyperframe] from_records error: {e}"); std::ptr::null_mut() }
    }
}

// ---------------------------------------------------------------------------
// Metadata
// ---------------------------------------------------------------------------

#[no_mangle]
pub extern "C" fn hf_frame_nrows(ptr: *const Frame) -> i64 {
    if ptr.is_null() { return -1; }
    unsafe { (*ptr).nrows as i64 }
}

#[no_mangle]
pub extern "C" fn hf_frame_ncols(ptr: *const Frame) -> i64 {
    if ptr.is_null() { return -1; }
    unsafe { (*ptr).schema.len() as i64 }
}

/// Returns JSON: {"col_name": "DType", ...}. Valid until next FFI call on this thread.
#[no_mangle]
pub extern "C" fn hf_frame_schema_json(ptr: *const Frame) -> *const c_char {
    if ptr.is_null() { return return_str("{}".into()); }
    let frame = unsafe { &*ptr };
    let map: Vec<String> = frame.schema.iter()
        .map(|(n, t)| format!("\"{}\":\"{}\"", n, t))
        .collect();
    return_str(format!("{{{}}}", map.join(",")))
}

// ---------------------------------------------------------------------------
// Aggregations
// ---------------------------------------------------------------------------

#[no_mangle]
pub extern "C" fn hf_frame_sum(ptr: *const Frame, col: *const c_char) -> f64 {
    if ptr.is_null() { return f64::NAN; }
    unsafe { &*ptr }.sum(cstr(col)).unwrap_or(f64::NAN)
}

#[no_mangle]
pub extern "C" fn hf_frame_mean(ptr: *const Frame, col: *const c_char) -> f64 {
    if ptr.is_null() { return f64::NAN; }
    unsafe { &*ptr }.mean(cstr(col)).unwrap_or(f64::NAN)
}

// ---------------------------------------------------------------------------
// Filtering
// ---------------------------------------------------------------------------

#[no_mangle]
pub extern "C" fn hf_frame_filter_gt(
    ptr: *const Frame, col: *const c_char, val: f64
) -> *mut Frame {
    if ptr.is_null() { return std::ptr::null_mut(); }
    Box::into_raw(Box::new(unsafe { &*ptr }.filter_gt(cstr(col), val)))
}

#[no_mangle]
pub extern "C" fn hf_frame_filter_eq_str(
    ptr: *const Frame, col: *const c_char, val: *const c_char
) -> *mut Frame {
    if ptr.is_null() { return std::ptr::null_mut(); }
    Box::into_raw(Box::new(unsafe { &*ptr }.filter_eq_str(cstr(col), cstr(val))))
}

// ---------------------------------------------------------------------------
// GroupBy and Sort
// ---------------------------------------------------------------------------

#[no_mangle]
pub extern "C" fn hf_frame_groupby_sum(
    ptr: *const Frame, group: *const c_char, agg: *const c_char
) -> *mut Frame {
    if ptr.is_null() { return std::ptr::null_mut(); }
    match unsafe { &*ptr }.groupby_sum(cstr(group), cstr(agg)) {
        Ok(f)  => Box::into_raw(Box::new(f)),
        Err(e) => { eprintln!("[hyperframe] groupby error: {e}"); std::ptr::null_mut() }
    }
}

#[no_mangle]
pub extern "C" fn hf_frame_sort_by(
    ptr: *const Frame, col: *const c_char, ascending: i32
) -> *mut Frame {
    if ptr.is_null() { return std::ptr::null_mut(); }
    match unsafe { &*ptr }.sort_by(cstr(col), ascending != 0) {
        Ok(f)  => Box::into_raw(Box::new(f)),
        Err(e) => { eprintln!("[hyperframe] sort error: {e}"); std::ptr::null_mut() }
    }
}
```

### Summary of Exported Functions

| Function | Purpose | Returns |
|----------|---------|---------|
| `hf_frame_free` | Free a Frame | void |
| `hf_frame_from_csv` | Load CSV file | `*mut Frame` |
| `hf_frame_from_records` | Load NDJSON bytes | `*mut Frame` |
| `hf_frame_nrows` | Row count | `i64` |
| `hf_frame_ncols` | Column count | `i64` |
| `hf_frame_schema_json` | Schema as JSON string | `*const c_char` |
| `hf_frame_sum` | Parallel column sum | `f64` |
| `hf_frame_mean` | Parallel column mean | `f64` |
| `hf_frame_filter_gt` | Filter rows (numeric >) | `*mut Frame` |
| `hf_frame_filter_eq_str` | Filter rows (text ==) | `*mut Frame` |
| `hf_frame_groupby_sum` | GroupBy + Sum | `*mut Frame` |
| `hf_frame_sort_by` | Sort by column | `*mut Frame` |

## Section 8: Cargo.toml and Build

```toml
[package]
name = "hyperframe-core"
version = "0.1.0"
edition = "2021"

[lib]
name = "hyperframe_core"
crate-type = ["cdylib"]

[dependencies]
rayon      = "1.10"
csv        = "1.3"
serde_json = "1.0"

[profile.release]
opt-level     = 3
lto           = true
codegen-units = 1
```

The `[lib]` section specifies `crate-type = ["cdylib"]`, which tells Cargo to produce a C-compatible dynamic library (`.so` on Linux, `.dylib` on macOS, `.dll` on Windows). The release profile enables maximum optimization (`opt-level = 3`), link-time optimization across all crate dependencies (`lto = true`), and single codegen unit (`codegen-units = 1`) for better whole-program optimization.

> **Advanced**: For production builds, you may want to add a custom global allocator (such as mimalloc or jemalloc) for better multi-threaded allocation performance. This is left as an exercise.

### Building

```bash
cargo build --release

# Verify exported symbols (Linux)
nm -D target/release/libhyperframe_core.so | grep "^[0-9a-f]* T hf_"

# Expected output:
# ... T hf_frame_filter_eq_str
# ... T hf_frame_filter_gt
# ... T hf_frame_free
# ... T hf_frame_from_csv
# ... T hf_frame_from_records
# ... T hf_frame_groupby_sum
# ... T hf_frame_mean
# ... T hf_frame_ncols
# ... T hf_frame_nrows
# ... T hf_frame_schema_json
# ... T hf_frame_sort_by
# ... T hf_frame_sum
```

The `T` prefix in the `nm` output indicates that these are exported text (code) symbols -- exactly what Python's `ctypes.CDLL` will look for.

### Module Structure

The Rust crate is organized as follows. Your `src/lib.rs` re-exports the public types:

```rust
// src/lib.rs
pub mod dtype;
pub mod column;
pub mod frame;
pub mod compute;
pub mod io;
pub mod ffi;

pub use dtype::DType;
pub use column::Column;
pub use frame::Frame;
```

## Section 9: Testing the Rust Core

Rust tests run with `cargo test`. Here are integration tests that verify our core operations:

```rust
// tests/integration_test.rs
use hyperframe_core::{Column, DType, Frame};

#[test]
fn test_filter_gt() {
    let frame = Frame {
        schema: vec![
            ("price".into(), DType::Float64),
            ("name".into(), DType::Text),
        ],
        columns: vec![
            Column::Float64(vec![10.0, 50.0, 200.0, 15.0]),
            Column::Text(vec!["a".into(), "b".into(), "c".into(), "d".into()]),
        ],
        nrows: 4,
    };

    let filtered = frame.filter_gt("price", 20.0);
    assert_eq!(filtered.nrows, 2);

    if let Column::Float64(prices) = &filtered.columns[0] {
        assert_eq!(prices, &[50.0, 200.0]);
    } else {
        panic!("Expected Float64 column");
    }
}

#[test]
fn test_filter_preserves_other_columns() {
    let frame = Frame {
        schema: vec![
            ("price".into(), DType::Float64),
            ("name".into(), DType::Text),
        ],
        columns: vec![
            Column::Float64(vec![10.0, 50.0, 200.0, 15.0]),
            Column::Text(vec!["a".into(), "b".into(), "c".into(), "d".into()]),
        ],
        nrows: 4,
    };

    let filtered = frame.filter_gt("price", 20.0);

    if let Column::Text(names) = &filtered.columns[1] {
        assert_eq!(names, &["b", "c"]);
    } else {
        panic!("Expected Text column");
    }
}

#[test]
fn test_groupby_sum() {
    let frame = Frame {
        schema: vec![
            ("region".into(), DType::Text),
            ("sales".into(), DType::Float64),
        ],
        columns: vec![
            Column::Text(vec!["North".into(), "South".into(), "North".into()]),
            Column::Float64(vec![100.0, 200.0, 150.0]),
        ],
        nrows: 3,
    };

    let result = frame.groupby_sum("region", "sales").unwrap();
    assert_eq!(result.nrows, 2);

    // BTreeMap gives sorted keys: North before South
    if let Column::Text(keys) = &result.columns[0] {
        assert_eq!(keys, &["North", "South"]);
    }
    if let Column::Float64(sums) = &result.columns[1] {
        assert_eq!(sums, &[250.0, 200.0]);
    }
}

#[test]
fn test_sum_and_mean() {
    let frame = Frame {
        schema: vec![("val".into(), DType::Float64)],
        columns: vec![Column::Float64(vec![10.0, 20.0, 30.0])],
        nrows: 3,
    };

    assert_eq!(frame.sum("val"), Some(60.0));
    assert_eq!(frame.mean("val"), Some(20.0));
}

#[test]
fn test_sort_by() {
    let frame = Frame {
        schema: vec![
            ("name".into(), DType::Text),
            ("score".into(), DType::Float64),
        ],
        columns: vec![
            Column::Text(vec!["c".into(), "a".into(), "b".into()]),
            Column::Float64(vec![30.0, 10.0, 20.0]),
        ],
        nrows: 3,
    };

    let sorted = frame.sort_by("score", true).unwrap();
    if let Column::Float64(scores) = &sorted.columns[1] {
        assert_eq!(scores, &[10.0, 20.0, 30.0]);
    }
    if let Column::Text(names) = &sorted.columns[0] {
        assert_eq!(names, &["a", "b", "c"]);
    }
}

#[test]
fn test_empty_frame() {
    let frame = Frame::empty(vec![
        ("id".into(), DType::Int64),
        ("name".into(), DType::Text),
    ]);
    assert_eq!(frame.nrows, 0);
    assert_eq!(frame.schema.len(), 2);
}
```

Run the tests:

```bash
cargo test
```

## Summary

In this chapter, you built a complete columnar data engine in Rust:

- **DType**: A simple type system covering Int64, Float64, Bool, and Text
- **Column**: Typed columnar storage using Rust's enum variants, each wrapping a contiguous `Vec<T>`
- **Frame**: The main data structure with schema, parallel columns, and row count invariant
- **Parallel compute**: Sum, mean, filter, groupby, and sort -- all using Rayon for multi-core execution
- **CSV and NDJSON I/O**: File loading with automatic type detection
- **12 FFI functions**: C-ABI exports using `Box::into_raw` for memory transfer and thread-local buffers for string returns

The Rust engine is complete. In the next chapter, we build the Python SDK that wraps it.

---

**Next**: [Chapter 3 - Bridging Rust and Python with ctypes](03-python-sdk.md) -- building the ctypes wrapper, DataFrame class, and memory lifecycle management.
