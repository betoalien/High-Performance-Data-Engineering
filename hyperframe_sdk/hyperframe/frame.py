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
