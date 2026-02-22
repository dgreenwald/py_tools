"""Tests for py_tools.text.core"""

import io
import os
import pytest

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
