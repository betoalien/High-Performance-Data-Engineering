---
title: Chapter 8 - Testing and Reliability
parent: High-Performance Data Engineering
nav_order: 9
has_children: false
description: Unit tests, integration tests, and property-based testing for hybrid Rust+Python systems
---

# Chapter 8: Testing and Reliability

A hybrid Rust+Python system has three layers that can fail independently: the Rust engine, the FFI boundary, and the Python SDK. Each layer requires its own testing strategy. In this chapter you will build a test suite that covers all three.

## The Testing Challenge in Hybrid Systems

Pure Rust code and pure Python code are each well-served by their own testing ecosystems. The hard part is the **FFI boundary**. Bugs at the boundary look like:

- Python receives a garbage value because a ctypes type signature is wrong
- A Rust function panics and the process crashes instead of raising a Python exception
- Memory is leaked because `hf_frame_free` is never called on an error path
- A function works on one platform but crashes on another because pointer widths differ

The principle: **test at every layer, but especially at the seams**.

---

## Section 1: Testing the Rust Engine

The Rust engine has unit tests built into the source files using `#[test]`. These run with `cargo test` and are completely independent of Python.

### Unit Tests for Core Data Structures

```rust
// src/frame.rs (within the module)

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_frame() -> Frame {
        Frame {
            schema: vec![
                ("price".into(), DType::Float64),
                ("qty".into(),   DType::Int64),
                ("region".into(), DType::Text),
            ],
            columns: vec![
                Column::Float64(vec![10.0, 20.0, 30.0, 40.0]),
                Column::Int64(vec![1, 2, 3, 4]),
                Column::Text(vec!["N".into(), "S".into(), "N".into(), "S".into()]),
            ],
            nrows: 4,
        }
    }

    #[test]
    fn test_sum() {
        let f = sample_frame();
        let s = f.sum("price").unwrap();
        assert!((s - 100.0).abs() < 1e-9, "expected 100.0, got {s}");
    }

    #[test]
    fn test_mean() {
        let f = sample_frame();
        let m = f.mean("price").unwrap();
        assert!((m - 25.0).abs() < 1e-9, "expected 25.0, got {m}");
    }

    #[test]
    fn test_filter_gt() {
        let f = sample_frame();
        let filtered = f.filter_gt("price", 15.0).unwrap();
        assert_eq!(filtered.nrows, 3); // 20, 30, 40 pass
    }

    #[test]
    fn test_filter_eq_str() {
        let f = sample_frame();
        let north = f.filter_eq_str("region", "N").unwrap();
        assert_eq!(north.nrows, 2);
    }

    #[test]
    fn test_groupby_sum() {
        let f = sample_frame();
        let g = f.groupby_sum("region", "price").unwrap();
        // N: 10+30=40, S: 20+40=60
        assert_eq!(g.nrows, 2);

        if let Column::Float64(vals) = &g.columns[1] {
            // BTreeMap sorts keys alphabetically: N before S
            assert!((vals[0] - 40.0).abs() < 1e-9);
            assert!((vals[1] - 60.0).abs() < 1e-9);
        } else {
            panic!("expected Float64 column");
        }
    }

    #[test]
    fn test_sort_ascending() {
        let f = sample_frame();
        let sorted = f.sort_by("price", true).unwrap();
        if let Column::Float64(vals) = &sorted.columns[0] {
            assert_eq!(vals[0], 10.0);
            assert_eq!(vals[3], 40.0);
        }
    }

    #[test]
    fn test_sort_descending() {
        let f = sample_frame();
        let sorted = f.sort_by("price", false).unwrap();
        if let Column::Float64(vals) = &sorted.columns[0] {
            assert_eq!(vals[0], 40.0);
            assert_eq!(vals[3], 10.0);
        }
    }

    #[test]
    fn test_column_length_invariant() {
        // All columns must have the same length as nrows
        let f = sample_frame();
        for col in &f.columns {
            assert_eq!(col.len(), f.nrows);
        }
    }

    #[test]
    fn test_unknown_column_returns_error() {
        let f = sample_frame();
        assert!(f.sum("nonexistent").is_err());
        assert!(f.filter_gt("nonexistent", 0.0).is_err());
    }
}
```

Run them with:

```bash
cd hyperframe-core/
cargo test
cargo test -- --nocapture    # See println! output
```

### Testing CSV Parsing

Create a test fixture file in `tests/fixtures/`:

```
# tests/fixtures/sample.csv
id,name,price,active
1,Widget A,9.99,true
2,Widget B,19.99,false
3,Widget C,4.99,true
```

```rust
// src/io.rs

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_csv_round_trip() {
        let csv = "id,price\n1,9.99\n2,19.99\n3,4.99\n";
        let mut f = NamedTempFile::new().unwrap();
        f.write_all(csv.as_bytes()).unwrap();

        let frame = read_csv(f.path().to_str().unwrap()).unwrap();
        assert_eq!(frame.nrows, 3);
        assert_eq!(frame.schema.len(), 2);

        // id column should auto-detect as Int64
        assert_eq!(frame.schema[0].1, DType::Int64);
        // price column should auto-detect as Float64
        assert_eq!(frame.schema[1].1, DType::Float64);
    }

    #[test]
    fn test_csv_missing_file_returns_error() {
        let result = read_csv("/nonexistent/path.csv");
        assert!(result.is_err());
    }
}
```

Add `tempfile` to `Cargo.toml` under `[dev-dependencies]`:

```toml
[dev-dependencies]
tempfile = "3"
```

### Property-Based Testing with Proptest

Property-based testing generates hundreds of random inputs and checks that invariants hold. Add `proptest` to `[dev-dependencies]`:

```toml
[dev-dependencies]
tempfile = "3"
proptest = "1"
```

```rust
// src/frame.rs

#[cfg(test)]
mod prop_tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        /// filter_gt followed by sum should always be <= original sum (for positive values)
        #[test]
        fn prop_filter_reduces_sum(
            prices in prop::collection::vec(0.0f64..1000.0, 1..500),
            threshold in 0.0f64..500.0,
        ) {
            let n = prices.len();
            let frame = Frame {
                schema: vec![("price".into(), DType::Float64)],
                columns: vec![Column::Float64(prices.clone())],
                nrows: n,
            };

            let filtered = frame.filter_gt("price", threshold).unwrap();
            let filtered_sum = filtered.sum("price").unwrap();
            let original_sum = frame.sum("price").unwrap();

            prop_assert!(filtered_sum <= original_sum + 1e-9);
        }

        /// sort ascending then descending should give reversed order
        #[test]
        fn prop_sort_reversible(
            values in prop::collection::vec(-1000.0f64..1000.0, 1..200),
        ) {
            let n = values.len();
            let frame = Frame {
                schema: vec![("v".into(), DType::Float64)],
                columns: vec![Column::Float64(values)],
                nrows: n,
            };

            let asc = frame.sort_by("v", true).unwrap();
            let desc = frame.sort_by("v", false).unwrap();

            if let (Column::Float64(a), Column::Float64(d)) =
                (&asc.columns[0], &desc.columns[0])
            {
                // ascending[i] == descending[n-1-i]
                for i in 0..n {
                    prop_assert!((a[i] - d[n - 1 - i]).abs() < 1e-12);
                }
            }
        }
    }
}
```

---

## Section 2: Testing the FFI Boundary

FFI boundary tests verify that Rust-allocated memory reaches Python correctly and is freed properly. These tests call the C-ABI functions directly, the same way the Python `ctypes` layer does.

```rust
// src/ffi.rs (bottom of file)

#[cfg(test)]
mod ffi_tests {
    use super::*;
    use std::ffi::CString;

    #[test]
    fn test_ffi_frame_from_records_and_free() {
        let ndjson = r#"{"x":1,"y":2.0}
{"x":3,"y":4.0}"#;
        let ptr = hf_frame_from_records(
            ndjson.as_ptr() as *const i8,
            ndjson.len(),
        );
        assert!(!ptr.is_null(), "hf_frame_from_records returned null");

        // Reading shape should work
        let nrows = hf_frame_nrows(ptr);
        assert_eq!(nrows, 2);

        // Free must not crash
        hf_frame_free(ptr);
        // ptr is now dangling -- do NOT use it after free
    }

    #[test]
    fn test_ffi_null_pointer_safety() {
        // All FFI functions must handle null gracefully
        let nrows = hf_frame_nrows(std::ptr::null_mut());
        assert_eq!(nrows, 0);

        let ncols = hf_frame_ncols(std::ptr::null_mut());
        assert_eq!(ncols, 0);

        // hf_frame_free(null) must be a no-op
        hf_frame_free(std::ptr::null_mut());
    }

    #[test]
    fn test_ffi_schema_json() {
        let ndjson = r#"{"name":"Alice","age":30}"#;
        let ptr = hf_frame_from_records(
            ndjson.as_ptr() as *const i8,
            ndjson.len(),
        );
        assert!(!ptr.is_null());

        let schema_ptr = hf_frame_schema_json(ptr);
        assert!(!schema_ptr.is_null());

        let schema_str = unsafe {
            std::ffi::CStr::from_ptr(schema_ptr).to_string_lossy().into_owned()
        };
        assert!(schema_str.contains("name"));
        assert!(schema_str.contains("age"));

        hf_frame_free(ptr);
    }
}
```

---

## Section 3: Testing the Python SDK

Python tests verify the full stack: FFI + Python wrapper + error handling. Use `pytest`.

```bash
pip install pytest
cd hyperframe_sdk/
pytest tests/ -v
```

### conftest.py: Shared Fixtures

```python
# hyperframe_sdk/tests/conftest.py
import pytest
import tempfile
import os
from hyperframe import DataFrame, read_csv


@pytest.fixture
def sample_df():
    """A small DataFrame for quick tests."""
    return DataFrame([
        {"region": "North", "product": "A", "revenue": 100.0, "qty": 5},
        {"region": "South", "product": "B", "revenue": 200.0, "qty": 10},
        {"region": "North", "product": "C", "revenue": 150.0, "qty": 7},
        {"region": "South", "product": "A", "revenue": 300.0, "qty": 15},
    ])


@pytest.fixture
def csv_file(tmp_path):
    """Write a temporary CSV and return its path."""
    content = "id,price,category\n1,9.99,A\n2,19.99,B\n3,4.99,A\n4,29.99,B\n"
    p = tmp_path / "test.csv"
    p.write_text(content)
    return str(p)
```

### test_dataframe.py: Core Functionality

```python
# hyperframe_sdk/tests/test_dataframe.py
import pytest
from hyperframe import DataFrame


class TestDataFrameCreation:
    def test_from_list_of_dicts(self, sample_df):
        assert sample_df.shape == (4, 4)

    def test_empty_list_raises(self):
        with pytest.raises(ValueError):
            DataFrame([])

    def test_non_list_raises(self):
        with pytest.raises(TypeError):
            DataFrame("not a list")

    def test_columns_property(self, sample_df):
        cols = sample_df.columns
        assert "region" in cols
        assert "revenue" in cols


class TestAggregations:
    def test_sum(self, sample_df):
        total = sample_df.sum("revenue")
        assert abs(total - 750.0) < 1e-9

    def test_mean(self, sample_df):
        avg = sample_df.mean("revenue")
        assert abs(avg - 187.5) < 1e-9

    def test_sum_unknown_column_raises(self, sample_df):
        with pytest.raises(RuntimeError):
            sample_df.sum("nonexistent")


class TestFiltering:
    def test_filter_gt(self, sample_df):
        high = sample_df.filter("revenue", ">", 150.0)
        assert high.shape[0] == 2  # 200 and 300 pass

    def test_filter_eq_str(self, sample_df):
        north = sample_df.filter("region", "==", "North")
        assert north.shape[0] == 2

    def test_filter_returns_new_object(self, sample_df):
        filtered = sample_df.filter("revenue", ">", 100.0)
        assert filtered is not sample_df

    def test_filter_unsupported_op_raises(self, sample_df):
        with pytest.raises(ValueError, match="Unsupported operator"):
            sample_df.filter("revenue", "<", 100.0)


class TestGroupBy:
    def test_groupby_sum_result_shape(self, sample_df):
        grouped = sample_df.groupby_sum("region", "revenue")
        # Two groups: North, South
        assert grouped.shape[0] == 2

    def test_groupby_sum_values(self, sample_df):
        grouped = sample_df.groupby_sum("region", "revenue")
        # North: 100+150=250, South: 200+300=500
        # BTreeMap sorts alphabetically: North first
        total = grouped.sum("revenue")
        assert abs(total - 750.0) < 1e-9


class TestSort:
    def test_sort_ascending(self, sample_df):
        sorted_df = sample_df.sort_by("revenue", ascending=True)
        assert sorted_df.shape == sample_df.shape

    def test_sort_descending(self, sample_df):
        sorted_df = sample_df.sort_by("revenue", ascending=False)
        assert sorted_df.shape == sample_df.shape

    def test_sort_preserves_row_count(self, sample_df):
        original_rows = sample_df.shape[0]
        sorted_df = sample_df.sort_by("revenue")
        assert sorted_df.shape[0] == original_rows
```

### test_io.py: CSV Loading

```python
# hyperframe_sdk/tests/test_io.py
import pytest
from hyperframe import read_csv


class TestReadCsv:
    def test_loads_file(self, csv_file):
        df = read_csv(csv_file)
        assert df.shape == (4, 3)

    def test_column_names(self, csv_file):
        df = read_csv(csv_file)
        assert "id" in df.columns
        assert "price" in df.columns
        assert "category" in df.columns

    def test_missing_file_raises(self):
        with pytest.raises(RuntimeError):
            read_csv("/no/such/file.csv")

    def test_sum_after_load(self, csv_file):
        df = read_csv(csv_file)
        total = df.sum("price")
        assert abs(total - 64.96) < 0.01
```

### test_memory.py: Memory Safety

```python
# hyperframe_sdk/tests/test_memory.py
"""
Memory safety tests -- verify there are no leaks or double-frees.

These tests rely on the OS not crashing. For more thorough checking,
run under Valgrind:
    valgrind --leak-check=full python -m pytest tests/test_memory.py
"""
import gc
import pytest
from hyperframe import DataFrame


class TestMemoryLifecycle:
    def test_del_is_called_on_gc(self):
        """Verify __del__ runs and clears _ptr."""
        df = DataFrame([{"x": 1.0}, {"x": 2.0}])
        ptr_value = df._ptr
        assert ptr_value is not None

        del df
        gc.collect()
        # After gc, the object is gone -- no way to assert _ptr is None,
        # but the test passing without a segfault confirms __del__ ran cleanly.

    def test_chained_operations_no_leak(self):
        """Intermediate frames from chaining must be freed by GC."""
        data = [{"revenue": float(i), "region": "N" if i % 2 == 0 else "S"}
                for i in range(1000)]
        df = DataFrame(data)

        # Each step creates an intermediate Frame in Rust
        result = (
            df.filter("revenue", ">", 100.0)
              .groupby_sum("region", "revenue")
        )
        assert result.shape[0] == 2

        del df, result
        gc.collect()
        # No crash = no double-free or use-after-free

    def test_multiple_references_same_pointer(self):
        """Two Python objects must NOT share the same Rust pointer."""
        df1 = DataFrame([{"x": 1.0}])
        # _from_ptr should be for internal use only --
        # users should never call it with an existing pointer
        # This test documents the rule: each DataFrame owns its pointer
        assert df1._ptr is not None
        ptr_before = df1._ptr
        del df1
        gc.collect()
        # ptr_before is now dangling -- the test just confirms no crash
```

---

## Section 4: Integration Tests

Integration tests verify the full pipeline from CSV ingestion through transformation to output.

```python
# hyperframe_sdk/tests/test_integration.py
import pytest
import tempfile
import os
from hyperframe import read_csv, DataFrame


def make_csv(tmp_path, rows=100):
    """Generate a CSV with controllable data."""
    lines = ["product,region,revenue,qty"]
    for i in range(rows):
        product = f"P{i % 10}"
        region = "North" if i % 3 == 0 else "South"
        revenue = float(10 + (i % 50))
        qty = (i % 20) + 1
        lines.append(f"{product},{region},{revenue},{qty}")
    p = tmp_path / "sales.csv"
    p.write_text("\n".join(lines) + "\n")
    return str(p)


class TestFullPipeline:
    def test_csv_to_groupby(self, tmp_path):
        path = make_csv(tmp_path, rows=100)
        df = read_csv(path)

        assert df.shape[0] == 100

        high_value = df.filter("revenue", ">", 30.0)
        by_region = high_value.groupby_sum("region", "revenue")

        assert by_region.shape[1] == 2  # region + revenue columns
        total = by_region.sum("revenue")
        assert total > 0

    def test_chained_pipeline_consistency(self, tmp_path):
        path = make_csv(tmp_path, rows=500)
        df = read_csv(path)

        # Full pipeline: filter -> groupby -> sort
        result = (
            df.filter("revenue", ">", 20.0)
              .groupby_sum("region", "revenue")
              .sort_by("revenue", ascending=False)
        )

        assert result.shape[0] == 2  # North and South

    def test_multiple_dataframes_independent(self, tmp_path):
        """Two DataFrames loaded from the same file are independent."""
        path = make_csv(tmp_path, rows=50)
        df1 = read_csv(path)
        df2 = read_csv(path)

        assert df1._ptr != df2._ptr  # separate Rust allocations
        assert df1.sum("revenue") == df2.sum("revenue")  # same data
```

---

## Section 5: Running the Full Test Suite

```bash
# Rust tests (engine + FFI boundary)
cd hyperframe-core/
cargo test
cargo test --release   # Also test optimized build

# Python tests (SDK + integration)
cd ../hyperframe_sdk/
pip install -e .
pip install pytest
pytest tests/ -v

# All tests with coverage
pip install pytest-cov
pytest tests/ --cov=hyperframe --cov-report=term-missing
```

### Expected Output

```
========================== test session starts ==========================
collected 32 items

tests/test_dataframe.py::TestDataFrameCreation::test_from_list_of_dicts PASSED
tests/test_dataframe.py::TestDataFrameCreation::test_empty_list_raises PASSED
tests/test_dataframe.py::TestAggregations::test_sum PASSED
tests/test_dataframe.py::TestAggregations::test_mean PASSED
tests/test_dataframe.py::TestFiltering::test_filter_gt PASSED
tests/test_dataframe.py::TestFiltering::test_filter_eq_str PASSED
tests/test_dataframe.py::TestGroupBy::test_groupby_sum_result_shape PASSED
tests/test_dataframe.py::TestSort::test_sort_ascending PASSED
tests/test_io.py::TestReadCsv::test_loads_file PASSED
tests/test_io.py::TestReadCsv::test_missing_file_raises PASSED
tests/test_memory.py::TestMemoryLifecycle::test_del_is_called_on_gc PASSED
tests/test_memory.py::TestMemoryLifecycle::test_chained_operations_no_leak PASSED
tests/test_integration.py::TestFullPipeline::test_csv_to_groupby PASSED
tests/test_integration.py::TestFullPipeline::test_chained_pipeline_consistency PASSED
...
========================== 32 passed in 0.84s ==========================
```

---

## Section 6: Testing on Multiple Platforms

Hybrid Rust+Python systems must be tested on every target platform. The main failure modes are:

| Issue | Cause | Detection |
|-------|-------|-----------|
| Wrong pointer width | `c_int` instead of `c_void_p` | Segfault on 64-bit Linux |
| Wrong calling convention | Platform ABI differences | Crash on Windows |
| Library not found | Wrong path detection | `ImportError` at import |
| Endianness | Arrow IPC byte order | Wrong values on big-endian |

### GitHub Actions Matrix

```yaml
# .github/workflows/test.yml
name: Test

on: [push, pull_request]

jobs:
  test:
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        python-version: ["3.10", "3.11", "3.12"]

    runs-on: ${{ matrix.os }}

    steps:
      - uses: actions/checkout@v4

      - name: Install Rust
        uses: actions-rs/toolchain@v1
        with:
          toolchain: stable

      - name: Build Rust library
        run: |
          cd hyperframe-core
          cargo build --release
          cargo test

      - name: Copy binary
        shell: bash
        run: |
          if [[ "${{ runner.os }}" == "Linux" ]]; then
            cp hyperframe-core/target/release/libhyperframe_core.so \
               hyperframe_sdk/hyperframe/libs/Linux/
          elif [[ "${{ runner.os }}" == "macOS" ]]; then
            ARCH=$(uname -m)
            cp hyperframe-core/target/release/libhyperframe_core.dylib \
               hyperframe_sdk/hyperframe/libs/Mac/libhyperframe_core-${ARCH}.dylib
          else
            cp hyperframe-core/target/release/hyperframe_core.dll \
               hyperframe_sdk/hyperframe/libs/Win/
          fi

      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install and test
        run: |
          pip install -e hyperframe_sdk/
          pip install pytest
          pytest hyperframe_sdk/tests/ -v
```

---

## Summary

In this chapter you built a three-layer test suite:

- **Rust unit tests** (`#[test]`): Verify each function's correctness in isolation
- **Proptest property tests**: Catch invariant violations on randomly generated inputs
- **FFI boundary tests**: Ensure null pointer safety and correct memory ownership
- **Python pytest suite**: Test the full SDK API with fixtures and parametrization
- **Integration tests**: Validate end-to-end pipeline correctness
- **CI matrix**: Catch platform-specific failures early

The most critical tests are the memory safety tests and the FFI null-pointer tests. A single unguarded null pointer dereference in production will crash the entire Python process with no Python traceback.

---

**Next**: [Chapter 9 - Profiling and Optimization](09-profiling.md) -- using flamegraphs and SIMD analysis to find and eliminate bottlenecks.
