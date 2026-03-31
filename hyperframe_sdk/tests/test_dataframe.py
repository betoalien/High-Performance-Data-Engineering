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
        assert grouped.shape[0] == 2

    def test_groupby_sum_total_preserved(self, sample_df):
        grouped = sample_df.groupby_sum("region", "revenue")
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
