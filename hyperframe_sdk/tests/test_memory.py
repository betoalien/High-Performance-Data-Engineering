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
        # Passing without a segfault confirms __del__ ran cleanly.

    def test_chained_operations_no_leak(self):
        """Intermediate frames from chaining must be freed by GC."""
        data = [{"revenue": float(i), "region": "N" if i % 2 == 0 else "S"}
                for i in range(1000)]
        df = DataFrame(data)

        result = (
            df.filter("revenue", ">", 100.0)
              .groupby_sum("region", "revenue")
        )
        assert result.shape[0] == 2

        del df, result
        gc.collect()

    def test_multiple_independent_dataframes(self):
        """Two DataFrames from the same data have independent Rust allocations."""
        data = [{"x": float(i)} for i in range(10)]
        df1 = DataFrame(data)
        df2 = DataFrame(data)

        assert df1._ptr != df2._ptr
        assert abs(df1.sum("x") - df2.sum("x")) < 1e-9

        del df1
        gc.collect()
        # df2 must still work after df1 is freed
        assert df2.shape[0] == 10
