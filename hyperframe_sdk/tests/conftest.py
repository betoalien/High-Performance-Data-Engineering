import pytest
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
