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
