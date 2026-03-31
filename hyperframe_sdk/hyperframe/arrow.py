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
