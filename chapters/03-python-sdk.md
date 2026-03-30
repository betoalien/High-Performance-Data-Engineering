---
title: Chapter 3 - Bridging the Gap
parent: High-Performance Data Engineering
nav_order: 4
has_children: false
description: FFI bindings, PyO3 alternatives, and designing Pythonic wrappers around Rust engines
---

# Chapter 3: Bridging the Gap (Python SDK & Interoperability)

This chapter covers connecting Python to our Rust core. We'll explore FFI (Foreign Function Interface) patterns, design Pythonic wrappers, and handle error propagation across the language boundary.

## Setting Up the FFI Layer

### Why ctypes Over PyO3?

Two main approaches exist for Python-Rust interop:

| Approach | Pros | Cons |
|----------|------|------|
| **PyO3** | Automatic type conversion, GIL management | Requires Python headers, tighter coupling |
| **ctypes** | Zero dependencies, pure C ABI, works with precompiled binaries | Manual type signatures, more boilerplate |

For our use case (distributing precompiled binaries with no Python dependencies), we use `ctypes`.

### Loading the Shared Library

```python
# wrapper.py

import ctypes
import os
import platform
import sys

# 1. Detect OS and select library path
system_os = sys.platform
current_dir = os.path.dirname(os.path.abspath(__file__))

if system_os == "win32":
    lib_name = "pardox-cpu-Windows-x64.dll"
    lib_folder = os.path.join(current_dir, "libs", "Win")
elif system_os == "linux":
    lib_name = "pardox-cpu-Linux-x64.so"
    lib_folder = os.path.join(current_dir, "libs", "Linux")
elif system_os == "darwin":
    lib_folder = os.path.join(current_dir, "libs", "Mac")
    machine_arch = platform.machine().lower()
    if "arm64" in machine_arch:
        lib_name = "pardox-cpu-MacOS-ARM64.dylib"
    elif "x86_64" in machine_arch:
        lib_name = "pardox-cpu-MacOS-Intel.dylib"
    else:
        raise OSError(f"Unsupported architecture: {machine_arch}")
else:
    raise OSError(f"Unsupported OS: {system_os}")

# 2. Build absolute path and load
lib_path = os.path.join(lib_folder, lib_name)

if not os.path.exists(lib_path):
    raise ImportError(f"PardoX Core binary not found at: {lib_path}")

# Add libs to LD_LIBRARY_PATH for transitive dependencies
_existing_ld = os.environ.get("LD_LIBRARY_PATH", "")
if lib_folder not in _existing_ld:
    os.environ["LD_LIBRARY_PATH"] = lib_folder + (":" + _existing_ld if _existing_ld else "")

try:
    lib = ctypes.CDLL(lib_path)
except OSError as e:
    raise ImportError(f"Failed to load PardoX Core: {e}")
```

### Defining C Type Signatures

```python
# Type aliases for clarity
c_void_p = ctypes.c_void_p
c_char_p = ctypes.c_char_p
c_longlong = ctypes.c_longlong
c_double = ctypes.c_double
c_int32 = ctypes.c_int32
c_size_t = ctypes.c_size_t

# =============================================================================
# Core API: Memory Management
# =============================================================================

# pardox_free_manager(*Manager) - Release Rust-allocated memory
lib.pardox_free_manager.argtypes = [c_void_p]
lib.pardox_free_manager.restype = None

# =============================================================================
# Core API: Data Loading
# =============================================================================

# pardox_load_manager_csv(path, schema_json, config_json) -> *Manager
lib.pardox_load_manager_csv.argtypes = [c_char_p, c_char_p, c_char_p]
lib.pardox_load_manager_csv.restype = c_void_p

# =============================================================================
# Core API: Joins
# =============================================================================

# pardox_hash_join(left, right, left_key, right_key) -> *Manager
lib.pardox_hash_join.argtypes = [c_void_p, c_void_p, c_char_p, c_char_p]
lib.pardox_hash_join.restype = c_void_p
```

## Designing a Pythonic Wrapper

### The DataFrame Class

We create a `DataFrame` class that wraps the raw pointer:

```python
# frame.py

class DataFrame:
    """
    A high-performance DataFrame backed by Rust memory.

    All computation happens in the Rust core; this class is a thin wrapper
    that provides a Pythonic interface.
    """

    @classmethod
    def _from_ptr(cls, ptr):
        """Internal method to create DataFrame from raw pointer."""
        obj = cls.__new__(cls)
        obj._ptr = ptr
        return obj

    def __init__(self, data=None, schema=None):
        """
        Initialize a DataFrame.

        Args:
            data: List of dicts, dict of lists, or None for empty
            schema: Optional schema specification
        """
        self._ptr = None

        # Handle dict input: convert to list of records
        if isinstance(data, dict):
            keys = list(data.keys())
            if not keys:
                data = []
            else:
                length = len(data[keys[0]])
                data = [{k: data[k][i] for k in keys} for i in range(length)]

        # Handle in-memory data (list of dicts)
        if isinstance(data, list):
            if not data:
                raise ValueError("Cannot create DataFrame from empty list.")

            # Convert to NDJSON (newline-delimited JSON)
            ndjson_str = "\n".join([json.dumps(record) for record in data])
            json_bytes = ndjson_str.encode('utf-8')

            # Call Rust FFI
            new_ptr = lib.pardox_read_json_bytes(json_bytes, len(json_bytes))

            if not new_ptr:
                raise RuntimeError("PardoX Core failed to ingest data.")

            self._ptr = new_ptr

        # Handle existing pointer (from native I/O functions)
        elif isinstance(data, (int, ctypes.c_void_p)) or \
             str(type(data)).find("LP_") != -1:
            if not data:
                raise ValueError("Null pointer received.")
            self._ptr = data

        elif data is not None:
            raise TypeError(f"Invalid input type: {type(data)}")

    def __del__(self):
        """Critical: Free Rust memory when Python object is garbage collected."""
        if self._ptr is not None:
            lib.pardox_free_manager(self._ptr)
            self._ptr = None
```

### Mixin Architecture for Organization

To keep the codebase maintainable, we split functionality across mixins:

```python
# frame.py

class DataFrame(
    VisualizationMixin,
    MetadataMixin,
    SelectionMixin,
    MutationMixin,
    WritersMixin,
    ExportMixin,
    MathMixin,
    GpuMixin,
    ReshapeMixin,
    TimeSeriesMixin,
    GroupByMixin,
    SqlMixin,
):
    """DataFrame with modular functionality via mixins."""
    pass
```

Each mixin handles a specific domain:

```python
# _ops.py (simplified example)

class MathMixin:
    """Arithmetic operations on columns."""

    def __add__(self, other):
        """Element-wise addition."""
        result_ptr = lib.pardox_series_add(
            self._ptr,
            self._key_column,
            other._ptr,
            other._key_column,
        )
        return DataFrame._from_ptr(result_ptr)

    def sum(self, column: str) -> float:
        """Aggregate sum of a column."""
        return lib.pardox_aggregate_sum(self._ptr, column.encode())


class SelectionMixin:
    """Row/column selection operations."""

    def __getitem__(self, key):
        """Column access: df['column_name']."""
        if isinstance(key, str):
            # Return column as Series
            return Series(self._ptr, key)
        elif isinstance(key, slice):
            # Row slicing: df[100:200]
            start = key.start or 0
            length = (key.stop or self.num_rows) - start
            ptr = lib.pardox_slice_manager(self._ptr, start, length)
            return DataFrame._from_ptr(ptr)

    def head(self, n: int = 5) -> "DataFrame":
        """Return first n rows."""
        ptr = lib.pardox_slice_manager(self._ptr, 0, n)
        return DataFrame._from_ptr(ptr)

    def tail(self, n: int = 5) -> "DataFrame":
        """Return last n rows."""
        ptr = lib.pardox_tail_manager(self._ptr, n)
        return DataFrame._from_ptr(ptr)


class MetadataMixin:
    """Schema and shape information."""

    @property
    def shape(self) -> tuple[int, int]:
        """Return (rows, columns) tuple."""
        n_rows = lib.pardox_get_row_count(self._ptr)
        schema_json = lib.pardox_get_schema_json(self._ptr)
        n_cols = len(json.loads(schema_json.decode()))
        return (n_rows, n_cols)

    @property
    def columns(self) -> list[str]:
        """Return list of column names."""
        schema_json = lib.pardox_get_schema_json(self._ptr)
        schema = json.loads(schema_json.decode())
        return list(schema.keys())

    def info(self) -> str:
        """Return DataFrame summary."""
        return lib.pardox_manager_to_ascii(self._ptr, 20).decode()
```

## Error Handling Across the Boundary

### Rust Side: Result to Error Codes

Rust functions that can fail return error codes or null pointers:

```rust
// api.rs

#[no_mangle]
pub extern "C" fn pardox_load_manager_csv(
    path: *const c_char,
    schema_json: *const c_char,
    config_json: *const c_char,
) -> *mut HyperBlockManager {
    // Validate inputs
    let Ok(path_str) = cstr_to_string(path) else {
        return std::ptr::null_mut();  // Invalid path pointer
    };

    // Attempt load
    match load_csv_internal(&path_str, schema_json, config_json) {
        Ok(manager) => Box::into_raw(Box::new(manager)),
        Err(e) => {
            eprintln!("[RUST ERROR] CSV load failed: {}", e);
            std::ptr::null_mut()  // Signal failure to Python
        }
    }
}
```

### Python Side: Checking for Failures

```python
def load_csv(path: str) -> DataFrame:
    """Load CSV file into DataFrame."""
    path_bytes = path.encode('utf-8')

    # Call Rust function
    ptr = lib.pardox_load_manager_csv(path_bytes, None, None)

    # Check for null pointer (failure)
    if not ptr:
        raise RuntimeError(f"Failed to load CSV: {path}")

    return DataFrame._from_ptr(ptr)
```

### Structured Error Reporting

For more detailed errors, we can use a thread-local error buffer:

```rust
// helper.rs

thread_local! {
    static ERROR_BUFFER: RefCell<CString> = RefCell::new(CString::new("").unwrap());
}

pub fn set_error(s: String) -> *const c_char {
    ERROR_BUFFER.with(|buf| {
        let c_str = CString::new(s).unwrap_or_default();
        let ptr = c_str.as_ptr();
        *buf.borrow_mut() = c_str;
        ptr
    })
}

#[no_mangle]
pub extern "C" fn pardox_get_last_error() -> *const c_char {
    ERROR_BUFFER.with(|buf| buf.borrow().as_ptr())
}
```

```python
# wrapper.py

def get_last_error() -> str:
    """Retrieve last error message from Rust."""
    ptr = lib.pardox_get_last_error()
    if ptr:
        return ctypes.c_char_p(ptr).value.decode()
    return "Unknown error"

# Usage in error handling
if not ptr:
    error_msg = get_last_error()
    raise RuntimeError(f"Rust operation failed: {error_msg}")
```

## Type Conversion Between Rust and Python

### Primitive Types

| Rust Type | ctypes Type | Python Type |
|-----------|-------------|-------------|
| `i64` | `c_longlong` | `int` |
| `f64` | `c_double` | `float` |
| `bool` | `c_bool` | `bool` |
| `&str` | `c_char_p` | `str` (encoded) |
| `*mut T` | `c_void_p` | opaque pointer |

### Complex Types: JSON Bridge

For complex data (schemas, configs), we use JSON as the interchange format:

```rust
// Rust side - parse JSON config
use serde_json;

#[derive(Deserialize)]
struct LoadConfig {
    delimiter: Option<char>,
    has_header: Option<bool>,
    null_string: Option<String>,
}

fn parse_config(json_ptr: *const c_char) -> Result<LoadConfig, String> {
    let json_str = cstr_to_string(json_ptr)?;
    serde_json::from_str(&json_str)
        .map_err(|e| format!("Invalid JSON config: {}", e))
}
```

```python
# Python side - serialize to JSON
import json

config = {
    "delimiter": ",",
    "has_header": True,
    "null_string": "NULL"
}
config_json = json.dumps(config).encode('utf-8')

ptr = lib.pardox_load_manager_csv(
    path_bytes,
    schema_json,
    config_json,
)
```

### Memory Ownership Contracts

Critical rule: **Whoever allocates must free**.

```rust
// Pattern 1: Rust allocates, Python frees via FFI
#[no_mangle]
pub extern "C" fn pardox_create_block(...) -> *mut HyperBlock {
    let block = HyperBlock::new(names, types);
    Box::into_raw(Box::new(block))  // Rust allocates
}

// Python must call pardox_free_manager() when done
```

```rust
// Pattern 2: Thread-local buffer (no free needed)
pub fn set_global_buffer(s: String) -> *const c_char {
    FFI_OUTPUT_BUFFER.with(|buf| {
        let c_str = CString::new(s).unwrap();
        let ptr = c_str.as_ptr();
        *buf.borrow_mut() = c_str;  // TLS keeps it alive
        ptr
    })
}
// Valid until next call on same thread - no free required
```

## Complete Example: Building the SDK

### Project Structure

```
pardox_sdk/
├── pardox/
│   ├── __init__.py      # Public API exports
│   ├── wrapper.py       # ctypes FFI layer
│   ├── frame.py         # DataFrame class
│   ├── series.py        # Series class
│   ├── io.py            # I/O functions
│   └── _ops.py          # Mixin classes
├── libs/
│   ├── Linux/
│   ├── Mac/
│   └── Win/
├── setup.py
└── README.md
```

### Public API (`__init__.py`)

```python
"""
PardoX SDK - High-Performance DataFrame Engine
"""

from .frame import DataFrame
from .series import Series
from .io import read_csv, read_json, read_sql, read_parquet

__version__ = "0.4.0"
__all__ = ["DataFrame", "Series", "read_csv", "read_json", "read_sql", "read_parquet"]
```

### I/O Functions (`io.py`)

```python
from .frame import DataFrame
from .wrapper import lib, c_char_p, c_void_p

def read_csv(path: str, schema: dict = None) -> DataFrame:
    """
    Load CSV file into a DataFrame.

    Args:
        path: Path to CSV file
        schema: Optional schema dict for type hints

    Returns:
        DataFrame backed by Rust memory
    """
    path_bytes = path.encode('utf-8')
    schema_json = json.dumps(schema).encode() if schema else None

    ptr = lib.pardox_load_manager_csv(path_bytes, schema_json, None)

    if not ptr:
        raise RuntimeError(f"Failed to load CSV: {path}")

    return DataFrame._from_ptr(ptr)


def read_json(path: str) -> DataFrame:
    """Load JSON/NDJSON file into DataFrame."""
    path_bytes = path.encode('utf-8')
    ptr = lib.pardox_load_manager_json(path_bytes)

    if not ptr:
        raise RuntimeError(f"Failed to load JSON: {path}")

    return DataFrame._from_ptr(ptr)


def read_sql(connection_string: str, query: str) -> DataFrame:
    """
    Execute SQL query and return results as DataFrame.

    Args:
        connection_string: Database connection (e.g., "postgresql://user:pass@host/db")
        query: SQL SELECT query

    Returns:
        DataFrame with query results
    """
    conn_bytes = connection_string.encode('utf-8')
    query_bytes = query.encode('utf-8')

    ptr = lib.pardox_scan_sql(conn_bytes, query_bytes)

    if not ptr:
        raise RuntimeError(f"SQL query failed: {query}")

    return DataFrame._from_ptr(ptr)
```

## Summary

This chapter covered:

- **FFI Setup**: Loading platform-specific shared libraries with `ctypes`
- **Type Signatures**: Defining `argtypes` and `restype` for safe calls
- **Pythonic Wrappers**: DataFrame class with mixins for organization
- **Memory Management**: `__del__` calling `pardox_free_manager()` to prevent leaks
- **Error Handling**: Null pointer returns and thread-local error buffers
- **Type Conversion**: JSON bridge for complex data structures
- **SDK Structure**: Organizing the Python package for distribution

In the next chapter, we'll build a complete ETL pipeline using our hybrid SDK.
