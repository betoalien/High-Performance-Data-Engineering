---
title: Chapter 2 - The Rust Core
parent: High-Performance Data Engineering
nav_order: 3
has_children: false
description: Building memory-safe data structures, concurrent processing, and compiling the Rust core
---

# Chapter 2: The Rust Core

This chapter covers building a high-performance data engine in Rust. We'll design memory-safe columnar data structures, implement parallel compute kernels, and configure the build for production use.

## Designing Memory-Safe Data Structures

### The HyperBlock: Atomic Storage Unit

The fundamental data structure in our engine is the `HyperBlock` — a columnar storage chunk optimized for cache-friendly access patterns.

```rust
// src/engine/core/hyperblock.rs

#[derive(Clone)]
pub struct ColumnData {
    pub dtype: HyperType,
    pub buffer: Vec<u8>,
    pub offsets: Option<Vec<u32>>,      // For variable-length types (Utf8, Json)
    pub null_bitmap: Option<Vec<u8>>,   // 1 = valid, 0 = null
    pub len: usize,
}

#[derive(Clone)]
pub struct HyperBlock {
    pub columns: Vec<ColumnData>,
    pub column_names: Vec<String>,
    pub nrows: usize,
}
```

### Why Columnar Storage?

Row-oriented storage (like Python lists of dicts) stores complete records together:

```
Row 0: {id: 1, name: "Alice", salary: 50000}
Row 1: {id: 2, name: "Bob", salary: 60000}
```

Columnar storage keeps each column's data contiguous:

```
id:     [1, 2, 3, 4, ...]
name:   ["Alice", "Bob", "Carol", ...]
salary: [50000, 60000, 55000, ...]
```

**Benefits for analytics:**

1. **Cache Efficiency**: Loading only needed columns reduces memory bandwidth
2. **SIMD Optimization**: Contiguous same-type data enables vectorized operations
3. **Compression**: Similar values compress better when stored together

### The HyperType System

Our internal type system bridges Rust and Python types:

```rust
// src/engine/core/types.rs

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HyperType {
    Int64,
    Float64,
    Boolean,
    Utf8,           // UTF-8 encoded strings
    Date,           // Days since epoch (i64)
    Timestamp,      // Microseconds since epoch (i64)
    Json,           // Raw UTF-8 with JSON semantics
    Decimal(u8),    // Fixed-point: value = raw_i64 / 10^scale
}

impl HyperType {
    pub fn fixed_size(&self) -> Option<usize> {
        match self {
            HyperType::Int64 | HyperType::Float64 |
            HyperType::Date | HyperType::Timestamp => Some(8),
            HyperType::Boolean => Some(1),
            HyperType::Utf8 | HyperType::Json => None,  // Variable length
            HyperType::Decimal(_) => Some(8),
        }
    }
}
```

### Memory Layout Details

For fixed-size types (Int64, Float64), data is stored as little-endian bytes:

```rust
// Converting i64 to buffer bytes
let value: i64 = 42;
buffer.extend_from_slice(&value.to_le_bytes());
// Buffer: [42, 0, 0, 0, 0, 0, 0, 0]

// Reading back
let bytes: [u8; 8] = buffer[0..8].try_into().unwrap();
let value = i64::from_le_bytes(bytes);
```

For variable-length types (Utf8), we use offset arrays:

```rust
// "Hello" + "World" stored with offsets
buffer:  [72, 101, 108, 108, 111, 87, 111, 114, 108, 100]
         // H    e    l    l    o    W    o    r    l    d
offsets: [0, 5, 10]  // Start positions for each string
```

## The HyperBlockManager: Partitioned Block Management

For large datasets, we partition data into multiple blocks:

```rust
// src/engine/core/blocks.rs

pub struct HyperBlockManager {
    pub schema: Arc<HyperSchema>,
    pub blocks: Vec<HyperBlock>,
    pub total_rows: usize,
}

impl HyperBlockManager {
    pub fn append_block(&mut self, block: HyperBlock) -> Result<(), String> {
        // Validate schema compatibility
        if self.schema.column_names.is_empty() {
            // First block defines the schema
            let cols: Vec<(String, HyperType)> = block
                .column_names
                .iter()
                .zip(block.columns.iter())
                .map(|(name, col)| (name.clone(), col.dtype))
                .collect();
            self.schema = Arc::new(HyperSchema::from_pairs(cols));
        } else {
            // Validate against existing schema
            let block_types: Vec<HyperType> = block.columns.iter().map(|c| c.dtype).collect();
            self.schema.validate_block(&block.column_names, &block_types)?;
        }

        self.total_rows += block.nrows;
        self.blocks.push(block);
        Ok(())
    }
}
```

### Why Partitioned Blocks?

1. **Incremental Loading**: Append data without reallocating entire dataset
2. **Parallel Processing**: Each block can be processed by a different thread
3. **Memory Efficiency**: Grow in chunks rather than doubling capacity

## Handling Data Transformations

### Compute Kernels

Compute operations are implemented as kernels that operate on `ColumnData`:

```rust
// src/engine/compute/kernels.rs

/// Fill null values in-place
pub fn kernel_fill_nulls(col: &mut ColumnData, fill_value: f64) {
    if col.dtype.is_numeric() {
        let nulls = match &col.null_bitmap {
            Some(n) => n,
            None => return,  // No nulls to fill
        };

        let fill_bytes = fill_value.to_le_bytes();
        for (i, &is_valid) in nulls.iter().enumerate() {
            if is_valid == 0 {
                let start = i * 8;
                col.buffer[start..start + 8].copy_from_slice(&fill_bytes);
            }
        }
    }
}

/// Round numeric column to specified decimals
pub fn kernel_round(col: &mut ColumnData, decimals: i32) {
    if let HyperType::Float64 = col.dtype {
        let multiplier = 10_f64.powi(decimals);
        for chunk in col.buffer.chunks_exact_mut(8) {
            let bytes: [u8; 8] = chunk.try_into().unwrap();
            let val = f64::from_le_bytes(bytes);
            let rounded = (val * multiplier).round() / multiplier;
            chunk.copy_from_slice(&rounded.to_le_bytes());
        }
    }
}
```

### Parallel Processing with Rayon

For large datasets, we use Rayon for data-parallel operations:

```rust
use rayon::prelude::*;

pub fn parallel_filter(blocks: &mut [HyperBlock], predicate: &[bool]) {
    blocks.par_iter_mut().for_each(|block| {
        *block = block.slice_rows(predicate);
    });
}
```

### Hash Joins

High-performance joins use hash tables:

```rust
// src/engine/sql/join.rs

pub fn hash_join_managers(
    left: &HyperBlockManager,
    right: &HyperBlockManager,
    left_key: &str,
    right_key: &str,
    join_type: JoinType,
) -> Result<HyperBlockManager, String> {
    // 1. Build hash table from right side
    let mut hash_map = std::collections::HashMap::new();
    for block in &right.blocks {
        // Build index on right_key column
    }

    // 2. Probe with left side
    let mut result_blocks = Vec::new();
    for block in &left.blocks {
        // Find matching rows using hash lookup
    }

    Ok(HyperBlockManager {
        schema: result_schema,
        blocks: result_blocks,
        total_rows: total_matched,
    })
}
```

## Concurrency at the System Level

### Global Allocator: MiMalloc

We use `mimalloc` for efficient multi-threaded memory allocation:

```rust
// src/lib.rs

use mimalloc::MiMalloc;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;
```

MiMalloc outperforms the default allocator in multi-threaded scenarios because:

- **Per-thread arenas**: Reduces lock contention
- **Size-class segregation**: Fast path for common allocation sizes
- **Cache-friendly**: Allocations from same thread stay close in memory

### Thread Pool Configuration

We configure thread pools based on detected CPU cores:

```rust
// src/helper.rs

use rayon::ThreadPoolBuilder;

lazy_static! {
    pub static ref COMPRESSION_POOL: ThreadPool = {
        let num_cores = num_cpus::get();

        // "Lookup table" strategy for optimal thread splits
        let reader_threads = match num_cores {
            0..=2 => 1,
            3..=5 => 2,
            6..=9 => 3,
            10..=14 => 4,
            _ => num_cores / 3,
        };

        let writer_threads = std::cmp::max(1, num_cores.saturating_sub(reader_threads));

        ThreadPoolBuilder::new()
            .num_threads(writer_threads)
            .build()
            .unwrap()
    };
}
```

### Thread-Local Storage for FFI Safety

When returning strings from FFI calls, we use thread-local storage to prevent use-after-free:

```rust
// src/helper.rs

thread_local! {
    static FFI_OUTPUT_BUFFER: RefCell<CString> = RefCell::new(CString::new("").unwrap());
}

pub fn set_global_buffer(s: String) -> *const c_char {
    FFI_OUTPUT_BUFFER.with(|buf| {
        let c_str = CString::new(s).unwrap_or_default();
        let ptr = c_str.as_ptr();
        *buf.borrow_mut() = c_str;  // Keep ownership in TLS
        ptr
    })
}
```

This ensures the string remains valid until the next FFI call on the same thread.

## Compiling the Rust Core

### Cargo Configuration

```toml
# Cargo.toml

[package]
name = "pardox-cpu"
version = "0.4.0"
edition = "2021"

[lib]
crate-type = ["cdylib"]  # Produces .so/.dylib/.dll

[profile.release]
opt-level = 3        # Maximum optimization
lto = "fat"          # Link-time optimization across crates
codegen-units = 1    # Single codegen unit for better optimization
panic = "abort"      # Smaller binaries, no unwind tables
```

### Build Commands

```bash
# Build optimized release library
cargo build --release

# Output locations by platform:
# Linux:   target/release/libpardox_cpu.so
# macOS:   target/release/libpardox_cpu.dylib
# Windows: target/release/pardox_cpu.dll
```

### Cross-Compilation

For building binaries for multiple platforms:

```bash
# Install cross-compilation toolchain
cargo install cross

# Build for Linux x64
cross build --release --target x86_64-unknown-linux-gnu

# Build for macOS Apple Silicon
cross build --release --target aarch64-apple-darwin

# Build for Windows x64
cross build --release --target x86_64-pc-windows-gnu
```

### Verifying the Build

```bash
# Check for compilation errors
cargo check

# Run tests
cargo test

# Lint with clippy
cargo clippy --release

# Verify exported symbols (Linux example)
nm -D target/release/libpardox_cpu.so | grep pardox_
```

## Summary

This chapter covered:

- **HyperBlock**: Columnar storage with contiguous memory layout
- **HyperType**: Internal type system bridging Rust and Python
- **HyperBlockManager**: Partitioned block management for incremental loading
- **Compute Kernels**: SIMD-friendly operations on column data
- **Concurrency**: MiMalloc allocator, Rayon thread pools, TLS for FFI
- **Build Configuration**: Release optimizations and cross-compilation

In the next chapter, we'll build the Python SDK that wraps this Rust core.
