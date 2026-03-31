import pytest
from hyperframe import read_csv, DataFrame


def make_csv(tmp_path, rows=100):
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

        assert by_region.shape[1] == 2
        total = by_region.sum("revenue")
        assert total > 0

    def test_chained_pipeline_consistency(self, tmp_path):
        path = make_csv(tmp_path, rows=500)
        df = read_csv(path)

        result = (
            df.filter("revenue", ">", 20.0)
              .groupby_sum("region", "revenue")
              .sort_by("revenue", ascending=False)
        )

        assert result.shape[0] == 2

    def test_multiple_dataframes_independent(self, tmp_path):
        path = make_csv(tmp_path, rows=50)
        df1 = read_csv(path)
        df2 = read_csv(path)

        assert df1._ptr != df2._ptr
        assert df1.sum("revenue") == df2.sum("revenue")
