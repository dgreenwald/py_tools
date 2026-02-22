"""Tests for py_tools.utilities (core and containers)."""

import numpy as np
import pytest

from py_tools.utilities.core import (
    as_list,
    split,
    split_str,
    split_list,
    any2,
    join_lists,
    check_duplicates,
    swap_all_axes,
    tic,
    toc,
    timer,
    log_if_pos,
    cartesian,
    cartesian_matrices,
    get_env,
)
from py_tools.utilities.containers import (
    MySet,
    MyDict,
    MyOrderedDict,
    PresetDict,
    UniqueList,
    replace_keys,
    replace_keys_items,
)


# ---------------------------------------------------------------------------
# core.py
# ---------------------------------------------------------------------------


class TestAsList:
    def test_non_list_wrapped(self):
        assert as_list(5) == [5]

    def test_list_returned_unchanged(self):
        assert as_list([1, 2]) == [1, 2]

    def test_none_wrapped(self):
        assert as_list(None) == [None]

    def test_string_wrapped(self):
        assert as_list("hello") == ["hello"]


class TestSplit:
    def test_split_1d(self):
        a = np.arange(6)
        parts = split(a, [2, 3, 1])
        assert len(parts) == 3
        assert np.array_equal(parts[0], [0, 1])
        assert np.array_equal(parts[1], [2, 3, 4])
        assert np.array_equal(parts[2], [5])

    def test_split_2d_rows(self):
        a = np.arange(9).reshape(3, 3)
        parts = split(a, [1, 2], axis=0)
        assert parts[0].shape == (1, 3)
        assert parts[1].shape == (2, 3)


class TestSplitStr:
    def test_basic(self):
        a, b = split_str("hello world", 5)
        assert a == "hello"
        assert b == " world"

    def test_converts_to_str(self):
        a, b = split_str(12345, 2)
        assert a == "12"
        assert b == "345"


class TestSplitList:
    def test_splits_correctly(self):
        first, second = split_list([1, 2, 3, 4], 2)
        assert first == [1, 2]
        assert second == [3, 4]

    def test_n_zero(self):
        first, second = split_list([1, 2, 3], 0)
        assert first == []
        assert second == [1, 2, 3]


class TestAny2:
    def test_match_found(self):
        assert any2([1, 2], [2, 3, 4]) is True

    def test_no_match(self):
        assert any2([5, 6], [1, 2, 3]) is False

    def test_empty_left(self):
        assert any2([], [1, 2]) is False


class TestJoinLists:
    def test_basic(self):
        assert join_lists([[1, 2], [3], [4, 5]]) == [1, 2, 3, 4, 5]

    def test_empty_lists_ignored(self):
        assert join_lists([[], [1], []]) == [1]

    def test_empty_outer(self):
        assert join_lists([]) == []


class TestCheckDuplicates:
    def test_no_duplicates_passes(self):
        check_duplicates([1, 2, 3])  # should not raise

    def test_duplicates_raise_with_message(self):
        with pytest.raises(Exception, match="Repeated variables detected"):
            check_duplicates([1, 2, 1])

    def test_empty_passes(self):
        check_duplicates([])  # should not raise


class TestSwapAllAxes:
    def test_no_swap_needed(self):
        a = np.arange(24).reshape(2, 3, 4)
        result = swap_all_axes(a, [0, 1, 2])
        assert np.array_equal(result, a)

    def test_transpose_2d(self):
        a = np.arange(6).reshape(2, 3)
        result = swap_all_axes(a, [1, 0])
        assert result.shape == (3, 2)
        assert np.array_equal(result, a.T)

    def test_3d_permutation(self):
        a = np.arange(24).reshape(2, 3, 4)
        # target_axes[i] = final destination of axis i
        # axis 0 (size 2) -> pos 1, axis 1 (size 3) -> pos 2, axis 2 (size 4) -> pos 0
        result = swap_all_axes(a, [1, 2, 0])
        assert result.shape == (4, 2, 3)


class TestTicToc:
    def test_tic_returns_float(self):
        t = tic()
        assert isinstance(t, float)

    def test_toc_returns_elapsed(self):
        t = tic()
        elapsed = toc(t)
        assert elapsed >= 0.0

    def test_elapsed_positive(self):
        import time as _time

        t = tic()
        _time.sleep(0.01)
        elapsed = toc(t)
        assert elapsed > 0.005


class TestTimer:
    def test_return_value_preserved(self):
        @timer
        def double(x):
            return x * 2

        result = double(7)
        assert result == 14

    def test_no_args(self):
        @timer
        def answer():
            return 42

        assert answer() == 42


class TestLogIfPos:
    def test_positive_array(self):
        x = np.array([1.0, np.e, np.e**2])
        result = log_if_pos(x)
        assert np.allclose(result, [0.0, 1.0, 2.0])

    def test_non_positive_returns_all_nan(self):
        # log_if_pos checks np.all(x > 0); if ANY element is non-positive,
        # the ENTIRE output is NaN (not element-wise).
        x = np.array([1.0, -1.0, 2.0])
        result = log_if_pos(x)
        assert np.all(np.isnan(result))

    def test_zeros_return_all_nan(self):
        # Zero is not positive, so the whole output is NaN.
        x = np.array([0.0, 1.0])
        result = log_if_pos(x)
        assert np.all(np.isnan(result))


class TestCartesian:
    def test_shape(self):
        result = cartesian(([1, 2], [3, 4, 5]))
        assert result.shape == (6, 2)

    def test_values(self):
        result = cartesian(([1, 2], [3, 4]))
        expected = np.array([[1, 3], [1, 4], [2, 3], [2, 4]])
        assert np.array_equal(result, expected)

    def test_three_arrays(self):
        result = cartesian(([1, 2], [3], [4, 5]))
        assert result.shape == (4, 3)


class TestCartesianMatrices:
    def test_output_shape(self):
        A = np.ones((2, 3))
        B = np.ones((4, 2))
        C = cartesian_matrices(A, B)
        assert C.shape == (8, 6, 2)

    def test_values_sampled(self):
        A = np.array([[1.0, 2.0]])
        B = np.array([[3.0, 4.0]])
        C = cartesian_matrices(A, B)
        assert C.shape == (1, 4, 2)
        assert np.array_equal(C[0, :, :], cartesian((A[0, :], B[0, :])))


class TestGetEnv:
    def test_returns_default_when_missing(self):
        val = get_env("NONEXISTENT_VAR_XYZ", "default_val")
        assert val == "default_val"

    def test_reads_set_env_var(self, monkeypatch):
        monkeypatch.setenv("MY_TEST_KEY", "hello")
        val = get_env("test_key", "fallback", prefix="MY")
        assert val == "hello"

    def test_prefix_underscore_added(self, monkeypatch):
        monkeypatch.setenv("PRE_KEY", "value")
        val = get_env("key", "fallback", prefix="pre")
        assert val == "value"

    def test_dtype_conversion(self, monkeypatch):
        monkeypatch.setenv("INT_VAR", "42")
        val = get_env("var", 0, prefix="INT", dtype=int)
        assert val == 42
        assert isinstance(val, int)

    def test_no_uppercase(self, monkeypatch):
        monkeypatch.setenv("myvar", "yes")
        val = get_env("myvar", "no", upper=False)
        assert val == "yes"

    def test_no_underscore_flag(self, monkeypatch):
        monkeypatch.setenv("PREKEY", "found")
        val = get_env("KEY", "missing", prefix="PRE", no_underscore=True)
        assert val == "found"


# ---------------------------------------------------------------------------
# containers.py
# ---------------------------------------------------------------------------


class TestMySet:
    def test_add(self):
        a = MySet([1, 2, 3])
        b = a + [3, 4, 5]
        assert isinstance(b, MySet)
        assert b == {1, 2, 3, 4, 5}

    def test_radd(self):
        a = MySet([1, 2])
        b = [2, 3] + a
        assert isinstance(b, MySet)
        assert b == {1, 2, 3}


class TestMyDict:
    def test_add_merges(self):
        a = MyDict(x=1, y=2)
        b = a + {"y": 99, "z": 3}
        assert isinstance(b, MyDict)
        assert b["x"] == 1
        assert b["y"] == 99  # right side wins
        assert b["z"] == 3

    def test_radd(self):
        a = MyDict(x=1)
        b = {"y": 2} + a
        assert isinstance(b, MyDict)
        assert b["x"] == 1
        assert b["y"] == 2


class TestMyOrderedDict:
    def test_add_preserves_order(self):
        from collections import OrderedDict

        a = MyOrderedDict([("a", 1), ("b", 2)])
        b = a + OrderedDict([("b", 99), ("c", 3)])
        assert isinstance(b, MyOrderedDict)
        assert list(b.keys()) == ["a", "b", "c"]
        assert b["b"] == 99  # right side wins


class TestPresetDict:
    def test_ignores_duplicate_key(self):
        d = PresetDict({"x": 1})
        d["x"] = 99
        assert d["x"] == 1

    def test_sets_new_key(self):
        d = PresetDict({"x": 1})
        d["y"] = 2
        assert d["y"] == 2

    def test_verbose_does_not_raise(self, capsys):
        d = PresetDict({"x": 1}, verbose=True)
        d["x"] = 99
        captured = capsys.readouterr()
        assert "x" in captured.out

    def test_update_skips_existing(self):
        d = PresetDict({"a": 1})
        d.update({"a": 2, "b": 3})
        assert d["a"] == 1
        assert d["b"] == 3

    def test_overwrite_item(self):
        d = PresetDict({"x": 1})
        d.overwrite_item("x", 99)
        assert d["x"] == 99

    def test_overwrite_update(self):
        d = PresetDict({"x": 1, "y": 2})
        d.overwrite_update({"x": 10, "z": 3})
        assert d["x"] == 10
        assert d["z"] == 3

    def test_kwargs_on_init(self):
        d = PresetDict(a=1, b=2)
        assert d["a"] == 1 and d["b"] == 2


class TestUniqueList:
    def test_deduplication_on_init(self):
        ul = UniqueList([1, 2, 2, 3])
        assert list(ul) == [1, 2, 3]

    def test_len(self):
        ul = UniqueList([1, 2, 3])
        assert len(ul) == 3

    def test_contains(self):
        ul = UniqueList([1, 2])
        assert 1 in ul
        assert 5 not in ul

    def test_iter(self):
        ul = UniqueList([10, 20, 30])
        assert list(ul) == [10, 20, 30]

    def test_add_returns_list(self):
        ul = UniqueList([1, 2])
        result = ul + [2, 3]
        assert result == [1, 2, 3]
        assert isinstance(result, list)

    def test_radd_returns_list(self):
        ul = UniqueList([2, 3])
        result = [1, 2] + ul
        assert result == [2, 3, 1]  # ul.data + new_uniques

    def test_iadd_preserves_type(self):
        ul = UniqueList([1, 2, 3])
        ul += [3, 4, 5]
        assert isinstance(ul, UniqueList)
        assert list(ul) == [1, 2, 3, 4, 5]

    def test_iadd_no_duplicates(self):
        ul = UniqueList([1, 2])
        ul += [2, 2, 3]
        assert list(ul) == [1, 2, 3]

    def test_empty_init(self):
        ul = UniqueList()
        assert list(ul) == []
        assert len(ul) == 0

    def test_repr(self):
        ul = UniqueList([1, 2])
        assert "UniqueList" in repr(ul)


class TestReplaceKeys:
    def test_renames_matching_keys(self):
        d = {"foo_1": "a", "foo_2": "b", "bar": "c"}
        replace_keys(d, r"foo_", "baz_")
        assert "baz_1" in d
        assert "baz_2" in d
        assert "foo_1" not in d

    def test_unmatched_key_unchanged(self):
        d = {"bar": "c"}
        replace_keys(d, r"foo_", "baz_")
        assert "bar" in d

    def test_returns_none(self):
        d = {"x": 1}
        result = replace_keys(d, "x", "y")
        assert result is None


class TestReplaceKeysItems:
    def test_renames_keys_and_string_values(self):
        d = {"foo_1": "val_1", "foo_2": "val_2"}
        replace_keys_items(d, r"_(\d+)", r"_num\1")
        assert d == {"foo_num1": "val_num1", "foo_num2": "val_num2"}

    def test_non_string_values_preserved(self):
        d = {"key_1": 42, "key_2": [1, 2]}
        replace_keys_items(d, r"_1", "_A")
        assert d["key_A"] == 42
        assert d["key_2"] == [1, 2]

    def test_empty_dict(self):
        d = {}
        replace_keys_items(d, "a", "b")
        assert d == {}
