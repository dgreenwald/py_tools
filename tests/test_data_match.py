"""Tests for py_tools.data.match"""

import numpy as np
import pytest

from py_tools.data.match import pairwise_match


class TestPairwiseMatch:
    def test_nearest_neighbor(self):
        treated = np.array([0.5])
        untreated = np.array([0.2, 0.45, 0.9])
        good_rows, min_ix = pairwise_match(treated, untreated)
        assert good_rows[0]
        assert min_ix[0] == 1  # 0.45 is closest to 0.5

    def test_all_rows_matched_without_caliper(self):
        treated = np.array([0.1, 0.5, 0.9])
        untreated = np.array([0.2, 0.6])
        good_rows, _ = pairwise_match(treated, untreated)
        assert np.all(good_rows)

    def test_caliper_excludes_distant(self):
        treated = np.array([0.5])
        untreated = np.array([0.9])  # distance 0.4 > caliper 0.3
        good_rows, _ = pairwise_match(treated, untreated, caliper=0.3)
        assert not np.any(good_rows)

    def test_caliper_keeps_close(self):
        treated = np.array([0.5])
        untreated = np.array([0.6])  # distance 0.1 < caliper 0.3
        good_rows, min_ix = pairwise_match(treated, untreated, caliper=0.3)
        assert good_rows[0]
        assert min_ix[0] == 0

    def test_replacement_false_raises(self):
        treated = np.array([0.5])
        untreated = np.array([0.4])
        with pytest.raises(Exception):
            pairwise_match(treated, untreated, replacement=False)

    def test_multiple_treated_same_match(self):
        # Both treated units closest to untreated[1] = 0.4
        treated = np.array([0.3, 0.5])
        untreated = np.array([0.1, 0.4, 0.8])
        good_rows, min_ix = pairwise_match(treated, untreated)
        assert min_ix[0] == 1
        assert min_ix[1] == 1
