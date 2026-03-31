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
