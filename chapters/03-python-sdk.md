---
title: Chapter 3 - Bridging Rust and Python with ctypes
parent: High-Performance Data Engineering
nav_order: 4
has_children: false
description: Building a Pythonic wrapper around the Rust engine using ctypes FFI
---

# Chapter 3: Bridging Rust and Python with ctypes

In Chapter 2, we built a Rust engine with 12 exported FFI functions. In this chapter, we build the Python SDK that wraps those functions into a clean, Pythonic API. By the end, you will be able to write:

```python
from hyperframe import DataFrame, read_csv

df = read_csv("sales.csv")
print(df.shape)                          # (1000000, 5)
print(df.sum("revenue"))                 # 4283917.50

high_value = df.filter("revenue", ">", 1000.0)
by_region  = high_value.groupby_sum("region", "revenue")
print(by_region)
```

All computation runs in Rust. Python holds only opaque pointers.

## Section 1: Why ctypes Over PyO3

There are two mainstream approaches for calling Rust from Python:

| Approach | Build-time deps | Distribution | Type safety | Boilerplate |
|----------|-----------------|--------------|-------------|-------------|
| **ctypes** | None (pure Python) | Precompiled binary | Manual | More |
| **PyO3** | Python headers + Rust toolchain | `maturin build` | Automatic | Less |

**ctypes** is the right choice when you want to ship precompiled binaries with zero build-time dependencies. Users install the Python package and get a working library immediately -- no Rust toolchain required. This is ideal for proprietary or commercial SDKs.

**PyO3** is better for open-source libraries where users compile from source, or when you want automatic GIL management and native Python type conversion. The trade-off is a tighter coupling between the Rust and Python build processes.

This course uses ctypes because it teaches the fundamentals of FFI explicitly, and because the distribution model (precompiled binaries per platform) is common in production data engineering.

## Section 2: The Wrapper Module

The wrapper module is the foundation of the SDK. It loads the platform-specific shared library and defines the type signatures for every FFI function.

```python
# hyperframe/wrapper.py
"""
ctypes FFI layer -- loads the compiled Rust library and defines all function signatures.
Import this module to get access to `lib` (the loaded ctypes.CDLL).
"""
import ctypes
import sys
import platform
from pathlib import Path


def _find_library() -> Path:
    """
    Locate the compiled Rust shared library relative to this file.

    The binary must be placed in:
        hyperframe/libs/Linux/libhyperframe_core.so         (Linux)
        hyperframe/libs/Mac/libhyperframe_core-arm64.dylib   (macOS ARM)
        hyperframe/libs/Mac/libhyperframe_core-x86_64.dylib  (macOS Intel)
        hyperframe/libs/Win/hyperframe_core.dll               (Windows)
    """
    base = Path(__file__).parent

    if sys.platform == "linux":
        lib_path = base / "libs" / "Linux" / "libhyperframe_core.so"
    elif sys.platform == "darwin":
        arch = platform.machine()
        name = ("libhyperframe_core-arm64.dylib" if arch == "arm64"
                else "libhyperframe_core-x86_64.dylib")
        lib_path = base / "libs" / "Mac" / name
    elif sys.platform == "win32":
        lib_path = base / "libs" / "Win" / "hyperframe_core.dll"
    else:
        raise OSError(f"Unsupported platform: {sys.platform}")

    if not lib_path.exists():
        raise ImportError(
            f"hyperframe-core binary not found at:\n  {lib_path}\n\n"
            "Build it first:\n"
            "  cd hyperframe-core/\n"
            "  cargo build --release\n"
            "  cp target/release/libhyperframe_core.so "
            "../hyperframe_sdk/hyperframe/libs/Linux/"
        )
    return lib_path


lib = ctypes.CDLL(str(_find_library()))

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
_ptr = ctypes.c_void_p    # opaque Rust pointer
_str = ctypes.c_char_p    # null-terminated C string

# ---------------------------------------------------------------------------
# Function signatures
# Defining these correctly is critical:
#   - Wrong argtypes -> silent data corruption
#   - Wrong restype  -> crashes or wrong values
# ---------------------------------------------------------------------------

# Memory management
lib.hf_frame_free.argtypes         = [_ptr]
lib.hf_frame_free.restype          = None

# Data loading
lib.hf_frame_from_csv.argtypes     = [_str]
lib.hf_frame_from_csv.restype      = _ptr

lib.hf_frame_from_records.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
lib.hf_frame_from_records.restype  = _ptr

# Metadata
lib.hf_frame_nrows.argtypes        = [_ptr]
lib.hf_frame_nrows.restype         = ctypes.c_int64

lib.hf_frame_ncols.argtypes        = [_ptr]
lib.hf_frame_ncols.restype         = ctypes.c_int64

lib.hf_frame_schema_json.argtypes  = [_ptr]
lib.hf_frame_schema_json.restype   = _str

# Aggregations
lib.hf_frame_sum.argtypes          = [_ptr, _str]
lib.hf_frame_sum.restype           = ctypes.c_double

lib.hf_frame_mean.argtypes         = [_ptr, _str]
lib.hf_frame_mean.restype          = ctypes.c_double

# Filtering
lib.hf_frame_filter_gt.argtypes    = [_ptr, _str, ctypes.c_double]
lib.hf_frame_filter_gt.restype     = _ptr

lib.hf_frame_filter_eq_str.argtypes = [_ptr, _str, _str]
lib.hf_frame_filter_eq_str.restype  = _ptr

# GroupBy and Sort
lib.hf_frame_groupby_sum.argtypes  = [_ptr, _str, _str]
lib.hf_frame_groupby_sum.restype   = _ptr

lib.hf_frame_sort_by.argtypes      = [_ptr, _str, ctypes.c_int32]
lib.hf_frame_sort_by.restype       = _ptr

# Arrow IPC (Chapter 4)
lib.hf_frame_to_arrow_ipc.argtypes = [_ptr, ctypes.POINTER(ctypes.c_size_t)]
lib.hf_frame_to_arrow_ipc.restype  = ctypes.POINTER(ctypes.c_uint8)

lib.hf_bytes_free.argtypes         = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t]
lib.hf_bytes_free.restype          = None
```

### Why Type Signatures Matter

Consider what happens if you get a type signature wrong:

```python
# WRONG: restype defaults to c_int (32-bit), but Rust returns *mut Frame (64-bit pointer)
lib.hf_frame_from_csv.restype = None  # forgot to set it!

ptr = lib.hf_frame_from_csv(b"data.csv")
# ptr is silently truncated to 32 bits on 64-bit systems
# Next call using ptr -> SEGFAULT or silent corruption
```

Always set both `argtypes` and `restype` for every function. This is the most common source of ctypes bugs.

## Section 3: The DataFrame Class

The DataFrame class is a thin Python wrapper around an opaque Rust pointer. No data is stored in Python -- all computation runs in the Rust engine.

```python
# hyperframe/frame.py
import json
import ctypes
from .wrapper import lib


class DataFrame:
    """
    A Rust-backed columnar DataFrame.

    Data is stored in Rust heap memory as contiguous typed arrays.
    Python holds only an opaque pointer; all computation runs in Rust.

    Memory lifecycle:
        1. Rust allocates a Frame and returns a *mut Frame pointer
        2. Python wraps it in this class
        3. When Python GC collects this object, __del__ calls hf_frame_free()
        4. Rust deallocates the Frame

    IMPORTANT: Never call hf_frame_free() manually -- let __del__ handle it.
    """

    def __init__(self, data: list | None = None):
        """
        Create a DataFrame from a list of dicts.

        Args:
            data: List of dicts, e.g. [{"name": "Alice", "age": 30}, ...]
        """
        self._ptr = None

        if data is None:
            return

        if not isinstance(data, list):
            raise TypeError(f"Expected list of dicts, got {type(data).__name__}")
        if not data:
            raise ValueError("Cannot create DataFrame from empty list")

        # Serialize to NDJSON and pass bytes to Rust
        ndjson = "\n".join(json.dumps(row) for row in data).encode("utf-8")
        self._ptr = lib.hf_frame_from_records(ndjson, len(ndjson))
        if not self._ptr:
            raise RuntimeError("Engine failed to create DataFrame from records")

    @classmethod
    def _from_ptr(cls, ptr: int) -> "DataFrame":
        """Wrap a raw Rust pointer. Internal use only."""
        obj = cls.__new__(cls)
        obj._ptr = ptr
        return obj

    def __del__(self):
        """Free Rust memory when this object is garbage collected."""
        if getattr(self, "_ptr", None):
            lib.hf_frame_free(self._ptr)
            self._ptr = None

    # -------------------------------------------------------------------
    # Shape and schema
    # -------------------------------------------------------------------

    @property
    def shape(self) -> tuple[int, int]:
        """(rows, columns)"""
        return (int(lib.hf_frame_nrows(self._ptr)),
                int(lib.hf_frame_ncols(self._ptr)))

    @property
    def columns(self) -> list[str]:
        """List of column names."""
        return list(self._schema().keys())

    @property
    def dtypes(self) -> dict[str, str]:
        """Column name -> dtype string mapping."""
        return self._schema()

    def _schema(self) -> dict[str, str]:
        raw = lib.hf_frame_schema_json(self._ptr)
        return json.loads(raw.decode("utf-8")) if raw else {}

    # -------------------------------------------------------------------
    # Aggregations
    # -------------------------------------------------------------------

    def sum(self, column: str) -> float:
        """Parallel column sum (all CPU cores via Rayon)."""
        return float(lib.hf_frame_sum(self._ptr, column.encode()))

    def mean(self, column: str) -> float:
        """Parallel column mean."""
        return float(lib.hf_frame_mean(self._ptr, column.encode()))

    # -------------------------------------------------------------------
    # Filtering
    # -------------------------------------------------------------------

    def filter(self, column: str, op: str, value) -> "DataFrame":
        """
        Filter rows by condition.

        Supported operators: '>' (numeric), '==' (string or numeric)

        Examples:
            df.filter("price", ">", 100.0)
            df.filter("region", "==", "North")
        """
        if op == ">":
            ptr = lib.hf_frame_filter_gt(
                self._ptr, column.encode(), float(value)
            )
        elif op == "==":
            ptr = lib.hf_frame_filter_eq_str(
                self._ptr, column.encode(), str(value).encode()
            )
        else:
            raise ValueError(
                f"Unsupported operator '{op}'. Use '>' or '=='"
            )

        if not ptr:
            raise RuntimeError(f"Filter failed (col='{column}', op='{op}')")
        return DataFrame._from_ptr(ptr)

    # -------------------------------------------------------------------
    # GroupBy
    # -------------------------------------------------------------------

    def groupby_sum(self, group_col: str, agg_col: str) -> "DataFrame":
        """
        Group by group_col and compute sum of agg_col.

        Returns a new DataFrame with columns: [group_col, {agg_col}_sum]
        Rows are sorted alphabetically by group_col.
        """
        ptr = lib.hf_frame_groupby_sum(
            self._ptr, group_col.encode(), agg_col.encode()
        )
        if not ptr:
            raise RuntimeError(
                f"groupby_sum failed (group='{group_col}', agg='{agg_col}')"
            )
        return DataFrame._from_ptr(ptr)

    # -------------------------------------------------------------------
    # Sorting
    # -------------------------------------------------------------------

    def sort_by(self, column: str, ascending: bool = True) -> "DataFrame":
        """Sort rows by column value."""
        ptr = lib.hf_frame_sort_by(
            self._ptr, column.encode(), 1 if ascending else 0
        )
        if not ptr:
            raise RuntimeError(f"sort_by failed on column '{column}'")
        return DataFrame._from_ptr(ptr)

    # -------------------------------------------------------------------
    # Arrow export (Chapter 4)
    # -------------------------------------------------------------------

    def to_pandas(self):
        """
        Convert to pandas DataFrame via Apache Arrow.
        Requires: pip install pyarrow pandas
        """
        from .arrow import to_pandas
        return to_pandas(self)

    def to_pyarrow(self):
        """Convert to PyArrow Table."""
        from .arrow import to_pyarrow_table
        return to_pyarrow_table(self)

    # -------------------------------------------------------------------
    # Display
    # -------------------------------------------------------------------

    def __repr__(self) -> str:
        rows, cols = self.shape
        col_names = self.columns
        return f"DataFrame({rows:,} rows x {cols} cols)\nColumns: {col_names}"
```

### Design Decisions

1. **No data in Python**: The `DataFrame` class stores only `_ptr` -- a `c_void_p` integer. There are no Python lists, no numpy arrays, no copies. This means creating a DataFrame costs zero Python memory beyond the object header.

2. **Lazy imports**: The `to_pandas()` method imports `arrow.py` at call time rather than at module level. This means pandas and pyarrow are optional dependencies -- users who don't need Arrow interop don't need to install them.

3. **Factory classmethod**: `_from_ptr` creates a DataFrame from an existing pointer without going through `__init__`. This is used by every method that returns a new DataFrame (filter, groupby, sort).

## Section 4: I/O Functions and Public API

```python
# hyperframe/io.py
from .frame import DataFrame
from .wrapper import lib


def read_csv(path: str) -> DataFrame:
    """
    Load a CSV file into a DataFrame.

    Column types are auto-detected by sampling the first 200 rows.
    Supported types: Int64, Float64, Bool, Text.
    """
    ptr = lib.hf_frame_from_csv(path.encode("utf-8"))
    if not ptr:
        raise RuntimeError(
            f"Failed to load '{path}'. "
            "Check: file exists, valid CSV, check stderr for details."
        )
    return DataFrame._from_ptr(ptr)
```

```python
# hyperframe/__init__.py
"""
HyperFrame -- a high-performance Rust-backed DataFrame for Python.

Built progressively throughout the "High-Performance Data Engineering" course.
"""
from .frame import DataFrame
from .io import read_csv

__version__ = "0.1.0"
__all__ = ["DataFrame", "read_csv"]
```

The public API is intentionally minimal. Users import `DataFrame` and `read_csv`; everything else is internal.

## Section 5: Memory Ownership -- The Most Important Rule

This section is the most critical in the entire course. Getting memory ownership wrong in FFI code produces crashes, corruption, or silent data loss. There are no exceptions or safety nets -- you must understand the rules.

### Rule 1: Rust Allocates, Rust Frees

Every `*mut Frame` returned by an `hf_*` function was created with:
```rust
Box::into_raw(Box::new(frame))
```

This allocates a `Frame` on the Rust heap and returns a raw pointer. Rust **gives up ownership** -- it will NOT free this memory. The caller (Python) is now responsible for eventually calling `hf_frame_free()`.

### Rule 2: Python's `__del__` is the Bridge

When Python's garbage collector collects a `DataFrame` object, `__del__` is called:
```python
def __del__(self):
    if getattr(self, "_ptr", None):
        lib.hf_frame_free(self._ptr)
        self._ptr = None
```

This is the ONLY place `hf_frame_free()` is called. Every allocation flows through this single point of deallocation.

### Rule 3: Double-Free is a Crash

If you call `hf_frame_free()` twice on the same pointer, Rust will attempt to deallocate already-freed memory. In debug builds, this triggers a panic. In release builds, it is undefined behavior -- anything from silent corruption to a segfault.

The `self._ptr = None` guard in `__del__` prevents this. After freeing, the pointer is set to None, so a second `__del__` call (which can happen in rare GC edge cases) is a no-op.

### Rule 4: Borrowed Pointers Are Not Owned

Functions like `hf_frame_sum(ptr, col)` take `*const Frame` -- a borrowed reference. They read the Frame but do not own it. **Never free a borrowed pointer.** Only free pointers returned by functions that explicitly transfer ownership (those returning `*mut Frame`).

### Memory Lifecycle Diagram

```
Python: df = read_csv("data.csv")
              |
              v
        hf_frame_from_csv("data.csv")
              |
              v [Rust heap]
        Box::new(Frame { ... })
        Box::into_raw(...)  -----------------> raw pointer
                                                    |
              +-------------------------------------+
              v
        DataFrame._ptr = raw pointer
              |
              |  (Python uses df.filter(), df.sum(), etc.)
              |  (Each filter/groupby creates a NEW Frame;
              |   old one is still alive until GC'd)
              |
              v [Python GC or del df]
        DataFrame.__del__()
              |
              v
        hf_frame_free(raw pointer)
              |
              v [Rust heap freed]
        drop(Box::from_raw(ptr))
```

### Common Pitfall: Chained Operations

```python
# This creates THREE Frames in Rust:
result = (
    df.filter("price", ">", 100)       # Frame #2 (df is Frame #1)
      .groupby_sum("region", "price")  # Frame #3
)

# Frame #2 (the intermediate filter result) has no Python reference.
# Python GC will collect it and call hf_frame_free().
# This is correct -- no leak, no double-free.
```

Each method returns a new DataFrame wrapping a new Rust pointer. The intermediate results are collected by Python's GC when they go out of scope. This is the correct behavior -- you do not need to manually free anything.

## Section 6: Error Handling Pattern

Errors cross the FFI boundary in two steps:

**Step 1 -- Rust side**: Functions that can fail return a null pointer and print diagnostics to stderr:

```rust
#[no_mangle]
pub extern "C" fn hf_frame_from_csv(path: *const c_char) -> *mut Frame {
    match crate::io::read_csv(cstr(path)) {
        Ok(f)  => Box::into_raw(Box::new(f)),
        Err(e) => {
            eprintln!("[hyperframe] read_csv error: {e}");
            std::ptr::null_mut()  // Signal failure
        }
    }
}
```

**Step 2 -- Python side**: The wrapper checks for null and raises a Python exception:

```python
ptr = lib.hf_frame_from_csv(path.encode("utf-8"))
if not ptr:
    raise RuntimeError(f"Failed to load '{path}'")
```

This two-level pattern is simple and reliable. The Rust side never panics across the FFI boundary (which would be undefined behavior). The Python side always checks for failure.

### Optional Improvement: Structured Error Buffer

For production use, you can add a thread-local error buffer that stores the last error message, retrievable from Python:

```rust
// Optional: structured error reporting
thread_local! {
    static LAST_ERROR: std::cell::RefCell<String> =
        std::cell::RefCell::new(String::new());
}

pub fn set_last_error(msg: &str) {
    LAST_ERROR.with(|e| *e.borrow_mut() = msg.to_string());
}

#[no_mangle]
pub extern "C" fn hf_last_error() -> *const c_char {
    LAST_ERROR.with(|e| return_str(e.borrow().clone()))
}
```

Python can then retrieve the error message:
```python
def _get_last_error() -> str:
    raw = lib.hf_last_error()
    return raw.decode("utf-8") if raw else "Unknown error"

# Usage:
if not ptr:
    raise RuntimeError(f"Operation failed: {_get_last_error()}")
```

## Section 7: Project Setup

### Directory Structure

```
hyperframe_sdk/
+-- hyperframe/
|   +-- __init__.py
|   +-- wrapper.py
|   +-- frame.py
|   +-- io.py
|   +-- arrow.py        (Chapter 4)
|   +-- libs/
|       +-- Linux/
|       |   +-- libhyperframe_core.so
|       +-- Mac/
|       |   +-- libhyperframe_core-arm64.dylib
|       |   +-- libhyperframe_core-x86_64.dylib
|       +-- Win/
|           +-- hyperframe_core.dll
+-- setup.py
```

Note that the `libs/` directory is inside the `hyperframe/` package, not at the top level. This is so that `package_data` can include the binaries when building a wheel.

### setup.py

```python
# setup.py
from setuptools import setup, find_packages

setup(
    name="hyperframe",
    version="0.1.0",
    packages=find_packages(),
    package_data={"hyperframe": ["libs/**/*"]},
    python_requires=">=3.10",
)
```

### Installation

```bash
# Build the Rust library first
cd hyperframe-core/
cargo build --release

# Copy binary to the SDK
cp target/release/libhyperframe_core.so \
   ../hyperframe_sdk/hyperframe/libs/Linux/

# Install the Python package in development mode
cd ../hyperframe_sdk/
pip install -e .

# Verify:
python -c "from hyperframe import DataFrame; print('OK')"
```

### Type Conversion Reference

| Rust Type | ctypes Type | Python Type | Notes |
|-----------|-------------|-------------|-------|
| `i64` | `c_int64` | `int` | 64-bit signed integer |
| `f64` | `c_double` | `float` | 64-bit IEEE 754 |
| `bool` | `c_bool` | `bool` | 1 byte |
| `*const c_char` | `c_char_p` | `bytes` | Null-terminated C string |
| `*mut Frame` | `c_void_p` | `int` | Opaque pointer |
| `usize` | `c_size_t` | `int` | Platform-width unsigned |

## Summary

In this chapter, you built the complete Python SDK for the Rust engine:

- **wrapper.py**: Platform-aware library loading and type-safe FFI signatures
- **frame.py**: DataFrame class with filter, groupby, sort, and aggregation methods
- **io.py**: CSV loading function
- **Memory lifecycle**: Rust allocates via `Box::into_raw`, Python frees via `__del__` calling `hf_frame_free`
- **Error handling**: Null pointer returns from Rust, RuntimeError in Python
- **Project structure**: Package layout with embedded platform binaries

The Python SDK is now functional. You can load CSV files, filter, aggregate, sort, and group data -- all running in Rust at native speed.

---

**Next**: [Chapter 4 - Apache Arrow: Zero-Copy Interoperability](04-apache-arrow.md) -- we add Arrow IPC export to transfer data from Rust to pandas without copying.
