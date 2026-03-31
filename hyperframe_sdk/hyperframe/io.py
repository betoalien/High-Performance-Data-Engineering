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
