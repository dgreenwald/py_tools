"""Tests for py_tools.text.core"""

import io
import os
import pytest
from types import SimpleNamespace

from py_tools.text.core import (
    Table,
    multicolumn,
    hstack,
    empty_table,
    shift_down,
    join_horizontal,
    join_vertical,
    open_latex,
    close_latex,
    regression_table,
    multi_regression_table,
    to_camel_case,
    write_values_tex,
    TSTRUT,
    BSTRUT,
)


# --- Table construction ---


class TestTableConstruction:
    def test_basic_construction(self):
        t = Table([["a", "b"], ["c", "d"]])
        assert t.n_rows() == 2
        assert t.n_cols == 2

    def test_default_alignment_is_cc(self):
        t = Table([["a", "b"]])
        assert t.alignment == "cc"

    def test_single_char_alignment_expanded(self):
        t = Table([["a", "b", "c"]], alignment="c")
        assert t.alignment == "ccc"

    def test_explicit_alignment_preserved(self):
        t = Table([["a", "b", "c"]], alignment="lcc")
        assert t.alignment == "lcc"

    def test_has_header_sets_hlines(self):
        t = Table([["h1", "h2"], ["v1", "v2"]], has_header=True)
        assert 0 in t.hlines

    def test_no_header_empty_hlines(self):
        t = Table([["a", "b"]], has_header=False)
        assert t.hlines == []

    def test_custom_hlines_respected(self):
        t = Table([["a", "b"], ["c", "d"]], hlines=[0, 1])
        assert t.hlines == [0, 1]

    def test_n_cols_explicit(self):
        t = Table([["a", "b", "c"]], n_cols=3)
        assert t.n_cols == 3

    def test_default_contents_empty(self):
        t = Table()
        assert t.contents == []
        assert t.n_cols == 0

    def test_tabu_alignment(self):
        t = Table([["a", "b"]], tabu=True)
        assert t.alignment == "X[c]X[c]"


# --- Table.n_rows ---


class TestTableNRows:
    def test_n_rows(self):
        t = Table([["a"], ["b"], ["c"]])
        assert t.n_rows() == 3

    def test_empty_rows(self):
        t = Table([])
        assert t.n_rows() == 0
        assert t.n_cols == 0


# --- Table.row ---


class TestTableRow:
    def test_valid_row(self):
        t = Table([["x", "y"], ["p", "q"]])
        assert t.row(0) == ["x", "y"]
        assert t.row(1) == ["p", "q"]

    def test_out_of_range_returns_empty(self):
        t = Table([["a", "b"]])
        assert t.row(5) == []


# --- Table.add_cline ---


class TestTableAddCline:
    def test_add_first_cline(self):
        t = Table([["a", "b"]])
        t.add_cline(1, 2, row=0)
        assert 0 in t.clines
        assert t.clines[0] == [(1, 2)]

    def test_add_multiple_clines_same_row(self):
        t = Table([["a", "b", "c"]])
        t.add_cline(1, 2, row=0)
        t.add_cline(2, 3, row=0)
        assert len(t.clines[0]) == 2


# --- Table.body ---


class TestTableBody:
    def test_body_contains_entries(self):
        t = Table([["Alpha", "Beta"]])
        body = t.body()
        assert "Alpha" in body
        assert "Beta" in body

    def test_body_float_formatting(self):
        t = Table([[1.23456]], floatfmt="5.2f")
        body = t.body()
        assert "1.23" in body

    def test_body_includes_tstrut_on_first_row(self):
        t = Table([["x", "y"], ["a", "b"]])
        body = t.body(include_tstrut=True)
        assert TSTRUT in body

    def test_body_no_tstrut_when_disabled(self):
        t = Table([["x", "y"]])
        body = t.body(include_tstrut=False)
        assert TSTRUT not in body

    def test_body_includes_bstrut_on_last_row(self):
        t = Table([["a", "b"], ["c", "d"]])
        body = t.body()
        assert BSTRUT in body

    def test_body_midrule_after_hline(self):
        t = Table([["h1", "h2"], ["v1", "v2"]], hlines=[0])
        body = t.body(booktabs=True)
        assert r"\midrule" in body

    def test_body_hline_non_booktabs(self):
        t = Table([["h1", "h2"], ["v1", "v2"]], hlines=[0])
        body = t.body(booktabs=False)
        assert r"\hline" in body


# --- Table.tabular ---


class TestTableTabular:
    def test_tabular_contains_tabular_env(self):
        t = Table([["a", "b"]])
        text = t.tabular()
        assert r"\begin{tabular}" in text
        assert r"\end{tabular}" in text

    def test_tabular_contains_toprule(self):
        t = Table([["a", "b"]])
        text = t.tabular(booktabs=True)
        assert r"\toprule" in text

    def test_tabular_hline_hline_non_booktabs(self):
        t = Table([["a", "b"]])
        text = t.tabular(booktabs=False)
        assert r"\hline \hline" in text

    def test_tabular_alignment_in_output(self):
        t = Table([["a", "b"]], alignment="lc")
        text = t.tabular()
        assert "{lc}" in text

    def test_tabu_uses_tabu_env(self):
        t = Table([["a", "b"]], tabu=True)
        text = t.tabular()
        assert r"\begin{tabu}" in text
        assert r"\end{tabu}" in text

    def test_super_header_included(self):
        t = Table([["a", "b"]])
        t.add_super_header("My Title")
        text = t.tabular()
        assert "My Title" in text


# --- Table.table ---


class TestTableTable:
    def test_table_wraps_in_table_env(self):
        t = Table([["a", "b"]])
        text = t.table()
        assert r"\begin{table}" in text
        assert r"\end{table}" in text

    def test_table_caption_above(self):
        t = Table([["a"]])
        text = t.table(caption="My Caption", caption_above=True)
        assert r"\caption{My Caption}" in text
        # Caption should appear before \end{center}
        assert text.index(r"\caption{My Caption}") < text.index(r"\end{center}")

    def test_table_caption_below(self):
        t = Table([["a"]])
        text = t.table(caption="My Caption", caption_above=False)
        assert r"\caption{My Caption}" in text
        assert text.index(r"\caption{My Caption}") > text.index(r"\end{center}")

    def test_table_notes(self):
        t = Table([["a"]])
        text = t.table(notes="Some footnote")
        assert r"\footnotesize{Some footnote}" in text

    def test_table_no_caption(self):
        t = Table([["a"]])
        text = t.table()
        assert r"\caption" not in text


# --- Table.write ---


class TestTableWrite:
    def test_write_tabular_to_file(self, tmp_path):
        out = tmp_path / "table.tex"
        t = Table([["a", "b"]])

        t.write(out)

        content = out.read_text()
        assert r"\begin{tabular}" in content
        assert r"\end{tabular}" in content

    def test_write_table_to_file(self, tmp_path):
        out = tmp_path / "table_float.tex"
        t = Table([["a"]])

        t.write(out, kind="table", caption="My Caption")

        content = out.read_text()
        assert r"\begin{table}" in content
        assert r"\caption{My Caption}" in content
        assert r"\end{table}" in content

    def test_write_invalid_kind_raises(self, tmp_path):
        out = tmp_path / "bad.tex"
        t = Table([["a"]])

        with pytest.raises(ValueError):
            t.write(out, kind="bad")


# --- multicolumn ---


class TestMulticolumn:
    def test_basic(self):
        result = multicolumn(3, "Header")
        assert result == r"\multicolumn{3}{c}{Header}"

    def test_single_col(self):
        result = multicolumn(1, "X")
        assert result == r"\multicolumn{1}{c}{X}"


# --- empty_table ---


class TestEmptyTable:
    def test_creates_correct_shape(self):
        t = empty_table(3, 4)
        assert t.n_rows() == 3
        assert t.n_cols == 4

    def test_default_alignment(self):
        t = empty_table(2, 3)
        assert t.alignment == "ccc"

    def test_custom_alignment(self):
        t = empty_table(2, 3, alignment="lcc")
        assert t.alignment == "lcc"

    def test_contents_filled_with_spaces(self):
        t = empty_table(2, 2)
        for row in t.contents:
            for cell in row:
                assert cell == " "


# --- hstack ---


class TestHstack:
    def test_combined_cols(self):
        t1 = Table([["a", "b"]])
        t2 = Table([["c", "d"]])
        merged = hstack([t1, t2])
        assert merged.n_cols == 4

    def test_combined_alignment(self):
        t1 = Table([["a"]], alignment="l")
        t2 = Table([["b"]], alignment="c")
        merged = hstack([t1, t2])
        assert merged.alignment == "lc"

    def test_row_count_uses_max(self):
        t1 = Table([["a"], ["b"]])
        t2 = Table([["c"]])
        merged = hstack([t1, t2])
        assert merged.n_rows() == 2

    def test_clines_offset_correctly(self):
        t1 = Table([["a", "b"]])
        t1.add_cline(1, 2, row=0)
        t2 = Table([["c", "d"]])
        merged = hstack([t1, t2])
        assert 0 in merged.clines
        assert (1, 2) in merged.clines[0]


# --- shift_down ---


class TestShiftDown:
    def test_new_row_prepended(self):
        t = Table([["a", "b"]])
        t2 = shift_down(t, ["header", "row"])
        assert t2.contents[0] == ["header", "row"]
        assert t2.n_rows() == 2

    def test_clines_shifted(self):
        t = Table([["a", "b"]])
        t.add_cline(1, 2, row=0)
        t2 = shift_down(t, [" ", " "])
        assert 1 in t2.clines

    def test_hlines_shifted(self):
        t = Table([["a", "b"], ["c", "d"]], hlines=[0])
        t2 = shift_down(t, [" ", " "])
        assert 1 in t2.hlines


# --- join_horizontal ---


class TestJoinHorizontal:
    def test_basic_join(self):
        t1 = Table([["a", "b"], ["c", "d"]])
        t2 = Table([["e", "f"], ["g", "h"]])
        result = join_horizontal([t1, t2], ["Header1", "Header2"])
        # Each table has 2 cols + 1 spacer col between them
        assert result.n_cols == 5

    def test_none_header_no_spacer(self):
        t1 = Table([["a", "b"]])
        t2 = Table([["c", "d"]])
        result = join_horizontal([t1, t2], [None, None])
        # No spacer inserted when header is None
        assert result.n_cols == 4

    def test_single_col_table_no_spacer(self):
        t1 = Table([["a"]])
        t2 = Table([["b"]])
        result = join_horizontal([t1, t2], ["H1", "H2"])
        # Single-col tables don't get spacers (n_cols == 1, not > 1)
        assert result.n_cols == 2


# --- join_vertical ---


class TestJoinVertical:
    def test_basic_vertical_join(self):
        t1 = Table([["a", "b"], ["c", "d"]])
        t2 = Table([["e", "f"], ["g", "h"]])
        result = join_vertical([t1, t2], ["Header1", "Header2"])
        # 1 header row + 2 data rows + 1 header row + 2 data rows = 6
        assert result.n_rows() == 6
        assert result.n_cols == 2

    def test_mismatched_cols_raises(self):
        t1 = Table([["a", "b"]])
        t2 = Table([["c", "d", "e"]])
        with pytest.raises(AssertionError):
            join_vertical([t1, t2], ["H1", "H2"])


# --- open_latex / close_latex ---


class TestOpenCloseLaTeX:
    def test_open_latex_writes_documentclass(self):
        buf = io.StringIO()
        open_latex(buf)
        content = buf.getvalue()
        assert r"\documentclass" in content
        assert r"\begin{document}" in content

    def test_close_latex_writes_end_document(self):
        buf = io.StringIO()
        close_latex(buf)
        assert r"\end{document}" in buf.getvalue()


# --- regression_table ---


class TestRegressionTable:
    def test_adds_significance_stars_by_default(self):
        results = SimpleNamespace(
            params=[1.0, 2.0, 3.0, 4.0],
            pvalues=[0.10, 0.05, 0.01, 0.20],
            HC0_se=[0.1, 0.2, 0.3, 0.4],
            rsquared_adj=0.9,
        )

        table = regression_table(results, var_names=["a", "b", "c", "d"])

        assert table.contents[1][:4] == ["1.000*", "2.000**", "3.000***", "4.000"]

    def test_can_disable_significance_stars(self):
        results = SimpleNamespace(
            params=[1.0],
            pvalues=[0.01],
            HC0_se=[0.1],
            rsquared_adj=0.9,
        )

        table = regression_table(results, var_names=["a"], add_stars=False)

        assert table.contents[1][0] == "1.000"

    def test_vertical_layout_adds_significance_stars(self):
        results = SimpleNamespace(
            params=[1.0],
            pvalues=[0.049],
            HC0_se=[0.1],
            rsquared_adj=0.9,
        )

        table = regression_table(results, vertical=True)

        assert table.contents[0][0] == "1.000**"

    def test_writes_tabular_file_when_path_provided(self, tmp_path):
        out = tmp_path / "regression_tabular.tex"
        results = SimpleNamespace(
            params=[1.0],
            pvalues=[0.01],
            HC0_se=[0.1],
            rsquared_adj=0.9,
        )

        table = regression_table(results, var_names=["a"], path=out)

        content = out.read_text()
        assert isinstance(table, Table)
        assert r"\begin{tabular}" in content
        assert "1.000***" in content

    def test_writes_table_file_when_requested(self, tmp_path):
        out = tmp_path / "regression_table.tex"
        results = SimpleNamespace(
            params=[1.0],
            pvalues=[0.20],
            HC0_se=[0.1],
            rsquared_adj=0.9,
        )

        regression_table(
            results,
            var_names=["a"],
            path=out,
            write_kind="table",
            write_kwargs={"caption": "Regression Output"},
        )

        content = out.read_text()
        assert r"\begin{table}" in content
        assert r"\caption{Regression Output}" in content

    def test_uses_result_param_names_when_var_names_omitted(self):
        class IndexedList(list):
            def __init__(self, values, index):
                super().__init__(values)
                self.index = index

        results = SimpleNamespace(
            params=IndexedList([1.0, 2.0], ["const", "x1"]),
            pvalues=[0.20, 0.01],
            HC0_se=[0.1, 0.2],
            rsquared_adj=0.9,
        )

        table = regression_table(results)

        assert table.contents[0][:2] == ["const", "x1"]

    def test_print_vars_can_use_result_variable_names(self):
        results = SimpleNamespace(
            params=[1.0, 2.0, 3.0],
            pvalues=[0.20, 0.01, 0.20],
            HC0_se=[0.1, 0.2, 0.3],
            rsquared_adj=0.9,
            model=SimpleNamespace(exog_names=["const", "x1", "x2"]),
        )

        table = regression_table(results, print_vars=["x1"])

        assert table.contents[0][0] == "x1"
        assert table.contents[1][0] == "2.000***"

    def test_var_labels_replace_result_names_for_display(self):
        results = SimpleNamespace(
            params=[1.0, 2.0],
            pvalues=[0.20, 0.01],
            HC0_se=[0.1, 0.2],
            rsquared_adj=0.9,
            model=SimpleNamespace(exog_names=["const", "x1"]),
        )

        table = regression_table(
            results,
            var_labels={"const": "Constant", "x1": "Main Effect"},
        )

        assert table.contents[0][:2] == ["Constant", "Main Effect"]

    def test_vertical_layout_respects_print_vars(self):
        results = SimpleNamespace(
            params=[1.0, 2.0],
            pvalues=[0.20, 0.01],
            HC0_se=[0.1, 0.2],
            rsquared_adj=0.9,
            model=SimpleNamespace(exog_names=["const", "x1"]),
        )

        table = regression_table(results, vertical=True, print_vars=["x1"])

        assert table.contents[0][0] == "2.000***"
        assert table.contents[1][0] == "(0.200)"
        assert len(table.contents) == 3

    def test_stat_headers_are_in_math_mode(self):
        results = SimpleNamespace(
            params=[1.0],
            pvalues=[0.20],
            HC0_se=[0.1],
            rsquared=0.8,
            rsquared_adj=0.75,
        )

        table = regression_table(
            results, var_names=["x1"], stats=["rsquared", "rsquared_adj"]
        )

        assert table.contents[0][1:] == ["$R^2$", "$\\bar{R}^2$"]

    def test_stat_labels_override_default_headers(self):
        results = SimpleNamespace(
            params=[1.0],
            pvalues=[0.20],
            HC0_se=[0.1],
            rsquared_adj=0.75,
        )

        table = regression_table(
            results,
            var_names=["x1"],
            stat_labels={"rsquared_adj": r"Adj. $R^2$"},
        )

        assert table.contents[0][1] == r"Adj. $R^2$"

    def test_nobs_is_formatted_as_integer_with_commas(self):
        results = SimpleNamespace(
            params=[1.0],
            pvalues=[0.20],
            HC0_se=[0.1],
            nobs=12345.0,
            rsquared_adj=0.75,
        )

        table = regression_table(
            results,
            var_names=["x1"],
            stats=["nobs", "rsquared_adj"],
        )

        assert table.contents[1][1] == "12,345"
        assert table.contents[0][1] == "$N$"


# --- to_camel_case ---


class TestToCamelCase:
    def test_basic(self):
        assert to_camel_case("hello_world") == "helloWorld"

    def test_no_underscore(self):
        assert to_camel_case("hello") == "hello"

    def test_multiple_underscores(self):
        assert to_camel_case("val_my_prefix") == "valMyPrefix"

    def test_empty_string(self):
        assert to_camel_case("") == ""


# --- write_values_tex ---


class TestWriteValuesTex:
    def test_creates_file(self, tmp_path):
        out = str(tmp_path / "values.tex")
        write_values_tex({"alpha": 0.05}, path=out)
        assert os.path.exists(out)

    def test_contains_explsyntaxon(self, tmp_path):
        out = str(tmp_path / "values.tex")
        write_values_tex({"k": "v"}, path=out)
        content = open(out).read()
        assert "\\ExplSyntaxOn" in content
        assert "\\ExplSyntaxOff" in content

    def test_key_value_present(self, tmp_path):
        out = str(tmp_path / "values.tex")
        write_values_tex({"mykey": "myval"}, path=out)
        content = open(out).read()
        assert "mykey" in content
        assert "myval" in content

    def test_prefix_applied(self, tmp_path):
        out = str(tmp_path / "values.tex")
        write_values_tex({"rate": "0.1"}, path=out, prefix="model")
        content = open(out).read()
        assert "model_rate" in content

    def test_custom_command_str(self, tmp_path):
        out = str(tmp_path / "values.tex")
        write_values_tex({"x": 1}, path=out, command_str="myCmd")
        content = open(out).read()
        assert "\\myCmd" in content

    def test_brace_escaping(self, tmp_path):
        out = str(tmp_path / "values.tex")
        write_values_tex({"k": "{braced}"}, path=out)
        content = open(out).read()
        assert r"\{braced\}" in content

    def test_prefix_in_command_when_add_prefix_true(self, tmp_path):
        out = str(tmp_path / "values.tex")
        write_values_tex({"k": "v"}, path=out, prefix="abc", add_prefix_to_command=True)
        content = open(out).read()
        assert "\\valAbc" in content

    def test_no_prefix_in_command_when_add_prefix_false(self, tmp_path):
        out = str(tmp_path / "values.tex")
        write_values_tex(
            {"k": "v"}, path=out, prefix="abc", add_prefix_to_command=False
        )
        content = open(out).read()
        assert "\\val " in content or "\\val\n" in content or "\\val #" in content


# --- multi_regression_table ---


def _make_results(**overrides):
    """Helper to build a mock results object."""
    defaults = dict(
        params=[1.0, 2.0],
        pvalues=[0.05, 0.01],
        HC0_se=[0.1, 0.2],
        tvalues=[10.0, 10.0],
        rsquared_adj=0.9,
        nobs=100.0,
        model=SimpleNamespace(exog_names=["x1", "x2"]),
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class TestMultiRegressionTable:
    def test_basic_two_models(self):
        r1 = _make_results()
        r2 = _make_results(
            params=[3.0, 4.0],
            pvalues=[0.20, 0.001],
            HC0_se=[0.3, 0.4],
            rsquared_adj=0.95,
        )
        table = multi_regression_table([r1, r2])

        # Header row
        assert table.contents[0] == ["", "(1)", "(2)"]
        # Two variables * 2 rows each + header + 1 stat = 6 rows
        assert table.n_rows() == 6
        assert table.n_cols == 3
        assert table.alignment == "lcc"

    def test_default_model_names(self):
        r1 = _make_results()
        r2 = _make_results()
        r3 = _make_results()
        table = multi_regression_table([r1, r2, r3])
        assert table.contents[0] == ["", "(1)", "(2)", "(3)"]

    def test_custom_model_names(self):
        r1 = _make_results()
        r2 = _make_results()
        table = multi_regression_table([r1, r2], model_names=["OLS", "IV"])
        assert table.contents[0] == ["", "OLS", "IV"]

    def test_significance_stars(self):
        r1 = _make_results(
            params=[1.0],
            pvalues=[0.009],
            HC0_se=[0.1],
            model=SimpleNamespace(exog_names=["x1"]),
        )
        table = multi_regression_table([r1])
        # Coeff row for x1
        assert "***" in table.contents[1][1]

    def test_no_stars_when_disabled(self):
        r1 = _make_results(
            params=[1.0],
            pvalues=[0.009],
            HC0_se=[0.1],
            model=SimpleNamespace(exog_names=["x1"]),
        )
        table = multi_regression_table([r1], add_stars=False)
        assert "***" not in table.contents[1][1]

    def test_different_variable_sets(self):
        r1 = _make_results(
            params=[1.0],
            pvalues=[0.05],
            HC0_se=[0.1],
            model=SimpleNamespace(exog_names=["x1"]),
        )
        r2 = _make_results(
            params=[2.0],
            pvalues=[0.01],
            HC0_se=[0.2],
            model=SimpleNamespace(exog_names=["x2"]),
        )
        table = multi_regression_table([r1, r2])

        # x1 row: present in r1, blank in r2
        assert table.contents[1][1] != ""  # r1 has x1
        assert table.contents[1][2] == ""  # r2 does not
        # x2 row: blank in r1, present in r2
        assert table.contents[3][1] == ""  # r1 does not have x2
        assert table.contents[3][2] != ""  # r2 has x2

    def test_var_labels(self):
        r1 = _make_results()
        table = multi_regression_table(
            [r1],
            var_labels={"x1": "Main Effect", "x2": "Control"},
        )
        assert table.contents[1][0] == "Main Effect"
        assert table.contents[3][0] == "Control"

    def test_explicit_var_names_ordering(self):
        r1 = _make_results()
        table = multi_regression_table([r1], var_names=["x2", "x1"])
        assert table.contents[1][0] == "x2"
        assert table.contents[3][0] == "x1"

    def test_print_vars_filters(self):
        r1 = _make_results()
        table = multi_regression_table([r1], print_vars=["x1"])
        # header + 1 var * 2 rows + 1 stat = 4
        assert table.n_rows() == 4
        assert table.contents[1][0] == "x1"

    def test_stats_default_rsquared_adj(self):
        r1 = _make_results(rsquared_adj=0.912)
        table = multi_regression_table([r1])
        # Last row should be the stat
        last_row = table.contents[-1]
        assert last_row[0] == "$\\bar{R}^2$"
        assert "0.912" in last_row[1]

    def test_stats_nobs_formatted_as_int(self):
        r1 = _make_results(nobs=12345.0)
        table = multi_regression_table([r1], stats=["nobs"])
        last_row = table.contents[-1]
        assert last_row[1] == "12,345"

    def test_multiple_stats(self):
        r1 = _make_results(rsquared_adj=0.9, nobs=500.0)
        table = multi_regression_table([r1], stats=["rsquared_adj", "nobs"])
        stat_rows = table.contents[-2:]
        labels = [row[0] for row in stat_rows]
        assert "$\\bar{R}^2$" in labels
        assert "$N$" in labels

    def test_tstat_mode(self):
        r1 = _make_results(
            params=[1.0],
            pvalues=[0.05],
            HC0_se=[0.1],
            tvalues=[10.0],
            model=SimpleNamespace(exog_names=["x1"]),
        )
        table = multi_regression_table([r1], tstat=True)
        # SE row should show t-stat value
        assert "(10.000)" in table.contents[2][1]

    def test_hlines_placed_correctly(self):
        r1 = _make_results()
        table = multi_regression_table([r1])
        # hline after header (0) and after last SE row
        assert 0 in table.hlines
        # Last SE row index: header(1) + 2 vars * 2 rows = 4, so index 4
        assert 4 in table.hlines

    def test_stat_labels_override(self):
        r1 = _make_results()
        table = multi_regression_table(
            [r1],
            stat_labels={"rsquared_adj": r"Adj. $R^2$"},
        )
        assert table.contents[-1][0] == r"Adj. $R^2$"

    def test_writes_file(self, tmp_path):
        out = tmp_path / "multi_reg.tex"
        r1 = _make_results()
        table = multi_regression_table([r1], path=out)
        content = out.read_text()
        assert isinstance(table, Table)
        assert r"\begin{tabular}" in content

    def test_se_in_parens(self):
        r1 = _make_results(
            params=[1.0],
            pvalues=[0.20],
            HC0_se=[0.123],
            model=SimpleNamespace(exog_names=["x1"]),
        )
        table = multi_regression_table([r1])
        assert table.contents[2][1] == "(0.123)"
