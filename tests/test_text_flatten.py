"""Tests for py_tools.text.flatten"""

import os
import pytest

from py_tools.text.flatten import (
    Command,
    get_next_argument,
    get_arguments,
    replace_content,
    replace_command,
    replace_commands_static,
    remove_comments,
    get_definitions,
    remove_command,
    remove_commands,
    remove_defn,
    remove_defns,
    replace_ref,
    replace_refs,
    get_commands,
    read_if_exists,
    remove_brackets,
    read_input_file,
    remove_unused_commands,
    replace_commands_dynamic,
    get_figure_labels,
    flatten_text,
    flatten,
)


# --- get_next_argument ---

class TestGetNextArgument:
    def test_simple_argument(self):
        arg, after = get_next_argument("{hello} rest")
        assert arg == "hello"
        assert after == " rest"

    def test_nested_argument(self):
        arg, after = get_next_argument("{a{b}c} rest")
        assert arg == "a{b}c"
        assert after == " rest"

    def test_no_opening_brace_raises(self):
        with pytest.raises(Exception):
            get_next_argument("no brace here")

    def test_whitespace_before_brace(self):
        arg, after = get_next_argument("  {value} tail")
        assert arg == "value"
        assert after == " tail"


# --- get_arguments ---

class TestGetArguments:
    def test_two_arguments(self):
        args, after = get_arguments("{first}{second} rest", 2)
        assert args == ["first", "second"]
        assert after == " rest"

    def test_zero_arguments(self):
        args, after = get_arguments("rest text", 0)
        assert args == []
        assert after == "rest text"


# --- replace_content ---

class TestReplaceContent:
    def test_no_args_command(self):
        cmd = Command("hello", 0, "world")
        result, after = replace_content(" rest", cmd)
        assert result == "world"
        assert after == " rest"

    def test_one_arg_command(self):
        cmd = Command("bold", 1, r"\textbf{#1}")
        result, after = replace_content("{text} tail", cmd)
        assert result == r"\textbf{text}"
        assert after == " tail"

    def test_two_arg_command(self):
        cmd = Command("frac", 2, r"\frac{#1}{#2}")
        result, after = replace_content("{a}{b} rest", cmd)
        assert result == r"\frac{a}{b}"
        assert after == " rest"

    def test_raises_for_ten_or_more_args(self):
        cmd = Command("too_many", 10, "")
        with pytest.raises(ValueError, match="fewer than 10"):
            replace_content("", cmd)


# --- replace_command ---

class TestReplaceCommand:
    def test_replaces_zero_arg_command(self):
        cmd = Command("greet", 0, "Hello")
        text = r"\greet world"
        result, replaced = replace_command(text, cmd)
        assert replaced
        assert "Hello" in result
        assert r"\greet" not in result

    def test_replaces_one_arg_command(self):
        cmd = Command("bold", 1, r"\textbf{#1}")
        text = r"\bold{word} rest"
        result, replaced = replace_command(text, cmd)
        assert replaced
        assert r"\textbf{word}" in result

    def test_no_match_returns_unchanged(self):
        cmd = Command("nonexistent", 0, "x")
        text = r"\other text"
        result, replaced = replace_command(text, cmd)
        assert not replaced
        assert result == text

    def test_replaces_multiple_occurrences(self):
        cmd = Command("X", 0, "Y")
        text = r"\X and \X again"
        result, _ = replace_command(text, cmd)
        assert "Y and Y again" == result

    def test_does_not_replace_partial_name(self):
        cmd = Command("foo", 0, "bar")
        text = r"\foobar is not \foo here"
        result, replaced = replace_command(text, cmd)
        assert replaced
        assert r"\foobar" in result  # partial match not replaced


# --- replace_commands_static ---

class TestReplaceCommandsStatic:
    def test_multiple_commands(self):
        cmds = [
            Command("A", 0, "Apple"),
            Command("B", 0, "Banana"),
        ]
        text = r"\A and \B "
        result = replace_commands_static(text, cmds)
        assert "Apple" in result
        assert "Banana" in result

    def test_empty_command_list(self):
        text = r"\something here"
        result = replace_commands_static(text, [])
        assert result == text

    def test_recursive_expansion(self):
        # \A expands to \B, \B expands to "final"
        cmds = [
            Command("A", 0, r"\B "),
            Command("B", 0, "final"),
        ]
        text = r"\A rest"
        result = replace_commands_static(text, cmds)
        assert "final" in result


# --- remove_comments ---

class TestRemoveComments:
    def test_removes_comment(self):
        text = r"text % this is a comment" + "\nmore text"
        result = remove_comments(text)
        assert "this is a comment" not in result
        assert "more text" in result

    def test_escaped_percent_preserved(self):
        text = r"price is \%50 % comment"
        result = remove_comments(text)
        assert r"\%50" in result

    def test_no_comment_unchanged(self):
        text = "plain text with no comment"
        result = remove_comments(text)
        assert result == text

    def test_multiple_comments(self):
        text = "line1 % c1\nline2 % c2\nline3"
        result = remove_comments(text)
        assert "line1" in result
        assert "line2" in result
        assert "line3" in result
        assert "c1" not in result
        assert "c2" not in result

    def test_comment_only_line(self):
        text = "% full comment line\nreal line"
        result = remove_comments(text)
        assert "full comment line" not in result
        assert "real line" in result


# --- get_definitions ---

class TestGetDefinitions:
    def test_finds_def(self):
        text = r"\def\myvar{42}"
        defs = get_definitions(text)
        assert len(defs) == 1
        assert defs[0].name == "myvar"
        assert defs[0].content == "42"
        assert defs[0].nargs == 0

    def test_finds_multiple_defs(self):
        text = "\\def\\alpha{a}\n\\def\\beta{b}\n"
        defs = get_definitions(text)
        names = [d.name for d in defs]
        assert "alpha" in names
        assert "beta" in names

    def test_no_defs(self):
        text = "plain text"
        defs = get_definitions(text)
        assert defs == []


# --- remove_command ---

class TestRemoveCommand:
    def test_removes_newcommand_zero_args(self):
        cmd = Command("myfunc", 0, "content")
        text = "\\newcommand{\\myfunc}{content}\nSome text \\myfunc here"
        result = remove_command(text, cmd)
        assert "\\newcommand" not in result
        assert "Some text" in result

    def test_removes_newcommand_with_args(self):
        cmd = Command("myfunc", 1, r"#1")
        text = "\\newcommand{\\myfunc}[1]{#1}\nUsage \\myfunc{x}"
        result = remove_command(text, cmd)
        assert "\\newcommand" not in result


# --- remove_commands ---

class TestRemoveCommands:
    def test_removes_multiple(self):
        cmds = [
            Command("foo", 0, "X"),
            Command("bar", 0, "Y"),
        ]
        text = "\\newcommand{\\foo}{X}\n\\newcommand{\\bar}{Y}\ntext"
        result = remove_commands(text, cmds)
        assert "\\newcommand{\\foo}" not in result
        assert "\\newcommand{\\bar}" not in result

    def test_empty_list(self):
        text = "unchanged"
        result = remove_commands(text, [])
        assert result == text


# --- remove_defn ---

class TestRemoveDefn:
    def test_removes_def(self):
        cmd = Command("mydef", 0, "value")
        text = "\\def\\mydef{value}\nSome text"
        result = remove_defn(text, cmd)
        assert "\\def\\mydef" not in result
        assert "Some text" in result


# --- remove_defns ---

class TestRemoveDefns:
    def test_removes_multiple_defns(self):
        defns = [
            Command("d1", 0, "v1"),
            Command("d2", 0, "v2"),
        ]
        text = "\\def\\d1{v1}\n\\def\\d2{v2}\ntext"
        result = remove_defns(text, defns)
        assert "\\def\\d1" not in result
        assert "\\def\\d2" not in result

    def test_empty_list(self):
        text = "unchanged"
        result = remove_defns(text, [])
        assert result == text


# --- replace_ref ---

class TestReplaceRef:
    def test_replaces_ref(self):
        text = r"See \ref{eq:1} for details"
        result = replace_ref(text, "eq:1", "3.14")
        assert "3.14" in result
        assert r"\ref{eq:1}" not in result

    def test_replaces_eqref_with_parens(self):
        text = r"Equation \eqref{eq:1} is important"
        result = replace_ref(text, "eq:1", "2")
        assert "(2)" in result
        assert r"\eqref{eq:1}" not in result

    def test_no_match_unchanged(self):
        text = r"text with \ref{other}"
        result = replace_ref(text, "eq:1", "5")
        assert result == text


# --- replace_refs ---

class TestReplaceRefs:
    def test_replaces_multiple(self):
        text = r"\ref{a} and \ref{b}"
        result = replace_refs(text, [("a", "1"), ("b", "2")])
        assert "1" in result
        assert "2" in result


# --- get_commands ---

class TestGetCommands:
    def test_finds_newcommand(self):
        text = "\\newcommand{\\myCmd}{content}\n"
        cmds = get_commands(text)
        assert len(cmds) == 1
        assert cmds[0].name == "myCmd"
        assert cmds[0].content == "content"
        assert cmds[0].nargs == 0

    def test_finds_newcommand_with_args(self):
        text = "\\newcommand{\\myCmd}[2]{#1 + #2}\n"
        cmds = get_commands(text)
        assert len(cmds) == 1
        assert cmds[0].nargs == 2

    def test_no_commands(self):
        text = "plain text"
        cmds = get_commands(text)
        assert cmds == []


# --- read_if_exists ---

class TestReadIfExists:
    def test_reads_existing_file(self, tmp_path):
        p = tmp_path / "test.txt"
        p.write_text("hello")
        result = read_if_exists(str(p))
        assert result == "hello"

    def test_returns_none_for_missing(self, tmp_path):
        result = read_if_exists(str(tmp_path / "nonexistent.txt"))
        assert result is None


# --- remove_brackets ---

class TestRemoveBrackets:
    def test_removes_outer_braces(self):
        result = remove_brackets("{hello}")
        assert result == "hello"

    def test_removes_nested_braces(self):
        result = remove_brackets("{outer{inner}end}")
        assert result == "outerinnerend"

    def test_no_braces_unchanged(self):
        result = remove_brackets("plain text")
        assert result == "plain text"


# --- read_input_file ---

class TestReadInputFile:
    def test_reads_file_directly(self, tmp_path):
        p = tmp_path / "myfile.tex"
        p.write_text("content")
        result = read_input_file(str(p))
        assert result == "content"

    def test_adds_tex_extension(self, tmp_path):
        p = tmp_path / "myfile.tex"
        p.write_text("with extension")
        # Pass path without extension
        result = read_input_file(str(tmp_path / "myfile"))
        assert result == "with extension"

    def test_returns_none_for_missing(self, tmp_path):
        result = read_input_file(str(tmp_path / "missing"))
        assert result is None


# --- remove_unused_commands ---

class TestRemoveUnusedCommands:
    def test_removes_unused(self):
        cmd = Command("unused", 0, "stuff")
        # Only one occurrence = definition itself, nothing uses it
        text = "\\newcommand{\\unused}{stuff}\nother text"
        result = remove_unused_commands(text, command_list=[cmd])
        assert "\\newcommand{\\unused}" not in result

    def test_keeps_used_command(self):
        cmd = Command("used", 0, "stuff")
        # Two occurrences: definition + usage
        text = "\\newcommand{\\used}{stuff}\n\\used here"
        result = remove_unused_commands(text, command_list=[cmd])
        assert "\\newcommand{\\used}" in result


# --- replace_commands_dynamic ---

class TestReplaceCommandsDynamic:
    def test_no_defs_returns_static_replaced(self):
        cmd = Command("A", 0, "Alpha")
        text = r"\A text"
        result = replace_commands_dynamic(text, [cmd])
        assert "Alpha" in result

    def test_preserves_def_not_in_names_to_replace(self):
        text = "\\def\\myvar{value}\nsome text"
        result = replace_commands_dynamic(text, [], names_to_replace=[])
        assert "\\def\\myvar" in result

    def test_expands_def_in_names_to_replace(self):
        text = "\\def\\myvar{expanded}\nprefix \\myvar suffix"
        result = replace_commands_dynamic(text, [], names_to_replace=["myvar"])
        assert "expanded" in result
        assert "prefix" in result

    def test_dynamic_update_replaces_body(self):
        # \def\X{hello} then use \X
        text = "\\def\\X{hello}\ntext \\X end"
        result = replace_commands_dynamic(text, [], names_to_replace=["X"])
        assert "hello" in result
        assert "text" in result


# --- get_figure_labels ---

class TestGetFigureLabels:
    def test_finds_figure_label(self):
        text = r"""
\begin{figure}
\includegraphics{fig1.pdf}
\label{fig:one}
\end{figure}
"""
        result = get_figure_labels(text)
        assert len(result) == 1
        assert result[0] == ("fig:one", "fig1.pdf")

    def test_finds_multiple_figures(self):
        text = r"""
\begin{figure}
\includegraphics{a.pdf}
\label{fig:a}
\end{figure}
\begin{figure}
\includegraphics{b.pdf}
\label{fig:b}
\end{figure}
"""
        result = get_figure_labels(text)
        assert len(result) == 2

    def test_figure_without_label_excluded(self):
        text = r"""
\begin{figure}
\includegraphics{nolabel.pdf}
\end{figure}
"""
        result = get_figure_labels(text)
        assert result == []


# --- flatten_text ---

class TestFlattenText:
    def test_plain_text_unchanged(self):
        text = "plain text no inputs"
        result = flatten_text(text)
        assert result == text

    def test_input_file_inlined(self, tmp_path):
        included = tmp_path / "part.tex"
        included.write_text("included content")
        text = r"\input{" + str(included) + "}"
        result = flatten_text(text)
        assert "included content" in result

    def test_missing_input_left_as_is(self, tmp_path):
        text = r"\input{/nonexistent/path/file.tex}"
        result = flatten_text(text)
        assert r"\input" in result

    def test_removes_comments_by_default(self):
        text = "text % comment\nmore"
        result = flatten_text(text, do_remove_comments_from_text=True)
        assert "comment" not in result

    def test_keeps_comments_when_disabled(self):
        text = "text % comment\nmore"
        result = flatten_text(text, do_remove_comments_from_text=False)
        assert "comment" in result

    def test_command_replacement_applied(self):
        cmd = Command("greet", 0, "Hello")
        text = r"\greet world"
        result = flatten_text(text, commands_to_replace=[cmd],
                              do_remove_comments_from_text=False)
        assert "Hello" in result


class TestFlatten:
    def test_flatten_does_not_change_cwd(self, tmp_path):
        root = tmp_path / "proj"
        root.mkdir()
        (root / "part.tex").write_text("included")
        (root / "main.tex").write_text("\\begin{document}\\input{part}\\end{document}")

        cwd_before = os.getcwd()
        result = flatten(infile=str(root / "main.tex"), outfile=None)
        cwd_after = os.getcwd()

        assert "included" in result
        assert cwd_after == cwd_before
