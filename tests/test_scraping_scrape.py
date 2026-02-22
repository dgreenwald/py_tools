"""Tests for py_tools.scraping.scrape"""

import importlib.util
import pytest
from unittest.mock import patch, MagicMock

from py_tools.scraping.scrape import get_soup, strip_html, stripHTML


# ---------------------------------------------------------------------------
# get_soup
# ---------------------------------------------------------------------------


class TestGetSoup:
    def _mock_response(
        self, status_code=200, content=b"<html><body><p>Hi</p></body></html>"
    ):
        resp = MagicMock()
        resp.status_code = status_code
        resp.content = content
        return resp

    def test_returns_soup_on_200(self):
        with (
            patch("requests.get", return_value=self._mock_response(200)),
            patch("time.sleep"),
        ):
            result = get_soup("http://example.com")
        assert result is not None
        assert result.find("p").text == "Hi"

    def test_returns_none_on_non_200(self, capsys):
        with (
            patch("requests.get", return_value=self._mock_response(404)),
            patch("time.sleep"),
        ):
            result = get_soup("http://example.com")
        assert result is None
        captured = capsys.readouterr()
        assert "404" in captured.out

    def test_prints_status_code_on_error(self, capsys):
        with (
            patch("requests.get", return_value=self._mock_response(500)),
            patch("time.sleep"),
        ):
            get_soup("http://example.com")
        assert "500" in capsys.readouterr().out

    def test_delay_is_applied(self):
        """The delay parameter must be passed to time.sleep."""
        with (
            patch("requests.get", return_value=self._mock_response(200)),
            patch("time.sleep") as mock_sleep,
        ):
            get_soup("http://example.com", delay=0.5)
        mock_sleep.assert_called_once_with(0.5)

    def test_default_delay_is_applied(self):
        """The default delay (1e-4) must also be passed to time.sleep."""
        with (
            patch("requests.get", return_value=self._mock_response(200)),
            patch("time.sleep") as mock_sleep,
        ):
            get_soup("http://example.com")
        mock_sleep.assert_called_once_with(1e-4)

    def test_delay_applied_even_on_error(self):
        """Delay should fire even when the server returns an error code."""
        with (
            patch("requests.get", return_value=self._mock_response(503)),
            patch("time.sleep") as mock_sleep,
        ):
            get_soup("http://example.com", delay=0.1)
        mock_sleep.assert_called_once_with(0.1)

    def test_parses_html_structure(self):
        html = (
            b"<html><head><title>T</title></head><body><h1>Heading</h1></body></html>"
        )
        with (
            patch("requests.get", return_value=self._mock_response(200, html)),
            patch("time.sleep"),
        ):
            soup = get_soup("http://example.com")
        assert soup.title.text == "T"
        assert soup.h1.text == "Heading"


# ---------------------------------------------------------------------------
# strip_html
# ---------------------------------------------------------------------------


class TestStripHtml:
    def test_returns_plain_text(self):
        html = b"<html><body><p>Hello World</p></body></html>"
        resp = MagicMock(status_code=200, content=html)
        with patch("requests.get", return_value=resp), patch("time.sleep"):
            result = strip_html("http://example.com")
        assert "Hello World" in result

    def test_returns_none_on_error(self):
        resp = MagicMock(status_code=500)
        with patch("requests.get", return_value=resp), patch("time.sleep"):
            result = strip_html("http://example.com")
        assert result is None

    def test_no_html_tags_in_output(self):
        html = b"<html><body><b>Bold</b> <i>Italic</i></body></html>"
        resp = MagicMock(status_code=200, content=html)
        with patch("requests.get", return_value=resp), patch("time.sleep"):
            result = strip_html("http://example.com")
        assert "<" not in result
        assert ">" not in result

    def test_stripHTML_is_alias_for_strip_html(self):
        """stripHTML must remain as a backward-compatible alias."""
        assert stripHTML is strip_html


# ---------------------------------------------------------------------------
# url_to_nltk
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    importlib.util.find_spec("nltk") is None,
    reason="nltk is an optional dependency",
)
class TestUrlToNltk:
    """Tests for url_to_nltk.  nltk.word_tokenize is mocked so that
    the punkt_tab corpus data resource does not need to be present."""

    def _mock_response(
        self,
        status_code=200,
        content=b"<html><body><p>The quick brown fox</p></body></html>",
    ):
        resp = MagicMock()
        resp.status_code = status_code
        resp.content = content
        return resp

    def test_returns_nltk_text_object(self):
        import nltk
        from py_tools.scraping.scrape import url_to_nltk

        tokens = ["The", "quick", "brown", "fox"]
        with (
            patch("requests.get", return_value=self._mock_response()),
            patch("time.sleep"),
            patch("nltk.word_tokenize", return_value=tokens),
        ):
            result = url_to_nltk("http://example.com")
        assert isinstance(result, nltk.Text)

    def test_returns_none_on_error(self):
        from py_tools.scraping.scrape import url_to_nltk

        resp = MagicMock(status_code=404)
        with patch("requests.get", return_value=resp), patch("time.sleep"):
            result = url_to_nltk("http://example.com")
        assert result is None

    def test_lower_true_lowercases_tokens(self):
        from py_tools.scraping.scrape import url_to_nltk

        html = b"<html><body><p>HELLO WORLD</p></body></html>"
        resp = MagicMock(status_code=200, content=html)
        # Capture the text passed to word_tokenize to confirm lowercasing.
        captured = {}

        def fake_tokenize(text):
            captured["text"] = text
            return text.split()

        with (
            patch("requests.get", return_value=resp),
            patch("time.sleep"),
            patch("nltk.word_tokenize", side_effect=fake_tokenize),
        ):
            url_to_nltk("http://example.com", lower=True)
        assert captured["text"] == captured["text"].lower()

    def test_lower_false_preserves_case(self):
        from py_tools.scraping.scrape import url_to_nltk

        html = b"<html><body><p>HELLO World</p></body></html>"
        resp = MagicMock(status_code=200, content=html)
        captured = {}

        def fake_tokenize(text):
            captured["text"] = text
            return text.split()

        with (
            patch("requests.get", return_value=resp),
            patch("time.sleep"),
            patch("nltk.word_tokenize", side_effect=fake_tokenize),
        ):
            url_to_nltk("http://example.com", lower=False)
        # Original mixed/uppercase text must be preserved.
        assert captured["text"] != captured["text"].lower()
