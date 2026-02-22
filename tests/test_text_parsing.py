"""Tests for py_tools.text.parsing"""

import pytest
from py_tools.text.parsing import match_to_bma, bma_search, expand_to_target, BmaMatch


# --- match_to_bma ---

class TestMatchToBma:
    def test_none_match_returns_unmatched(self):
        bma = match_to_bma(None, "hello world")
        assert not bma.matched
        assert bma.before == "hello world"
        assert bma.middle == ""
        assert bma.after == ""
        assert bma.match is None

    def test_valid_match_splits_correctly(self):
        import re
        text = "foo BAR baz"
        match = re.search(r"BAR", text)
        bma = match_to_bma(match, text)
        assert bma.matched
        assert bma.before == "foo "
        assert bma.middle == "BAR"
        assert bma.after == " baz"
        assert bma.match is match

    def test_match_at_start(self):
        import re
        text = "XYZ rest"
        match = re.search(r"XYZ", text)
        bma = match_to_bma(match, text)
        assert bma.before == ""
        assert bma.middle == "XYZ"
        assert bma.after == " rest"

    def test_match_at_end(self):
        import re
        text = "start END"
        match = re.search(r"END", text)
        bma = match_to_bma(match, text)
        assert bma.before == "start "
        assert bma.middle == "END"
        assert bma.after == ""


# --- bma_search ---

class TestBmaSearch:
    def test_found_pattern(self):
        bma = bma_search(r"\d+", "abc 123 def")
        assert bma.matched
        assert bma.middle == "123"
        assert bma.before == "abc "
        assert bma.after == " def"

    def test_not_found_pattern(self):
        bma = bma_search(r"\d+", "no digits here")
        assert not bma.matched
        assert bma.before == "no digits here"
        assert bma.middle == ""
        assert bma.after == ""

    def test_returns_bma_match_namedtuple(self):
        bma = bma_search(r"x", "axb")
        assert isinstance(bma, BmaMatch)

    def test_extra_flags_passed_through(self):
        import re
        bma = bma_search(r"^hello", "HELLO world", re.IGNORECASE | re.MULTILINE)
        assert bma.matched


# --- expand_to_target ---

class TestExpandToTarget:
    def test_simple_parentheses(self):
        content, bracket, after = expand_to_target("a + b) rest", left_bracket="(")
        assert content == "a + b"
        assert bracket == ")"
        assert after == " rest"

    def test_simple_curly_braces(self):
        content, bracket, after = expand_to_target("hello} world", left_bracket="{")
        assert content == "hello"
        assert bracket == "}"
        assert after == " world"

    def test_simple_square_brackets(self):
        content, bracket, after = expand_to_target("item] rest", left_bracket="[")
        assert content == "item"
        assert bracket == "]"
        assert after == " rest"

    def test_nested_parens(self):
        # Outer content includes the nested paren
        content, bracket, after = expand_to_target("(inner) outer) end", left_bracket="(")
        assert content == "(inner) outer"
        assert bracket == ")"
        assert after == " end"

    def test_nested_curly_braces(self):
        content, bracket, after = expand_to_target("{nested} outer} end", left_bracket="{")
        assert content == "{nested} outer"
        assert bracket == "}"
        assert after == " end"

    def test_no_content_before_close(self):
        content, bracket, after = expand_to_target(") trailing", left_bracket="(")
        assert content == ""
        assert bracket == ")"
        assert after == " trailing"

    def test_invalid_bracket_raises(self):
        with pytest.raises(Exception):
            expand_to_target("text", left_bracket="<")

    def test_no_closing_bracket_raises(self):
        with pytest.raises(Exception):
            expand_to_target("no closing bracket at all", left_bracket="(")

    def test_content_with_special_chars(self):
        content, bracket, after = expand_to_target(r"\alpha + \beta} rest", left_bracket="{")
        assert content == r"\alpha + \beta"
        assert bracket == "}"
        assert after == " rest"
