#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 29 21:59:19 2025

@author: dan
"""

from collections import namedtuple
import re

BmaMatch = namedtuple("BmaMatch", ["before", "middle", "after", "match", "matched"])


def match_to_bma(match, text):
    """Convert a regex match object into a ``BmaMatch`` named tuple.

    Parameters
    ----------
    match : re.Match or None
        A compiled regular-expression match object, or ``None`` when no match
        was found.
    text : str
        The original string that was searched.

    Returns
    -------
    BmaMatch
        Named tuple with fields:

        ``before`` : str
            Text preceding the match (or the whole string when ``match`` is
            ``None``).
        ``middle`` : str
            The matched substring (empty string when ``match`` is ``None``).
        ``after`` : str
            Text following the match (empty string when ``match`` is
            ``None``).
        ``match`` : re.Match or None
            The original match object.
        ``matched`` : bool
            ``True`` when ``match`` is not ``None``.
    """
    if match is None:
        before = text
        middle = ""
        after = ""
        matched = False
    else:
        before = text[: match.start()]
        middle = text[match.start() : match.end()]
        after = text[match.end() :]
        matched = True

    return BmaMatch(before, middle, after, match, matched)


def bma_search(pattern, text, *args, **kwargs):
    """Search for a pattern in text and return a ``BmaMatch`` named tuple.

    A thin wrapper around :func:`re.search` that splits the result into
    before/middle/after segments via :func:`match_to_bma`.

    Parameters
    ----------
    pattern : str or re.Pattern
        Regular-expression pattern to search for.
    text : str
        String to search within.
    *args
        Additional positional arguments forwarded to :func:`re.search`.
    **kwargs
        Additional keyword arguments forwarded to :func:`re.search`.

    Returns
    -------
    BmaMatch
        Named tuple as described in :func:`match_to_bma`.
    """
    match = re.search(pattern, text, *args, **kwargs)
    return match_to_bma(match, text)


def expand_to_target(
    text,
    target=None,
    depth=1,
    target_depth=1,
    left_bracket="(",
    right_bracket=")",
    before_text="",
):
    """Recursively expand text until a target token is found at the correct nesting depth.

    Scans ``text`` character by character (via regex) tracking bracket nesting
    depth.  When the ``target`` token is encountered at ``target_depth``, the
    function returns the text accumulated before the target, the target itself,
    and the remaining text after the target.

    Parameters
    ----------
    text : str
        Remaining text to scan.
    target : str, optional
        Token to search for.  Defaults to the closing bracket that corresponds
        to ``left_bracket``.
    depth : int, optional
        Current nesting depth at the start of this call.  Defaults to ``1``.
    target_depth : int or None, optional
        Nesting depth at which the target should be recognized.  Pass ``None``
        to match the target at any depth.  Defaults to ``1``.
    left_bracket : str, optional
        Opening bracket character: one of ``'('``, ``'['``, or ``'{'``.
        Defaults to ``'('``.
    right_bracket : str, optional
        Closing bracket character.  Automatically derived from
        ``left_bracket``; the parameter is kept for API compatibility but is
        overridden internally.
    before_text : str, optional
        Accumulated text from previous recursive calls.  Defaults to ``''``.

    Returns
    -------
    before : str
        Text accumulated before the matched target token.
    middle : str
        The matched target token.
    after : str
        Remaining text after the target token.

    Raises
    ------
    Exception
        If ``left_bracket`` is not one of the supported bracket characters, or
        if no matching token can be found.
    """
    if left_bracket == "(":
        right_bracket = ")"
    elif left_bracket == "[":
        right_bracket = "]"
    elif left_bracket == "{":
        right_bracket = "}"
    else:
        raise Exception

    if target is None:
        target = right_bracket

    pattern = (
        rf"({re.escape(left_bracket)}|{re.escape(right_bracket)}|{re.escape(target)})"
    )

    bma = bma_search(pattern, text)
    if not bma.matched:
        raise Exception

    before_text += bma.before

    if target in bma.match.group(0):
        # We found the target
        if (target_depth is None) or (depth == target_depth):
            # We are at the right depth level, return
            return before_text, bma.middle, bma.after
        else:
            # We are at the wrong depth
            new_depth = depth

    # We found a bracket
    if left_bracket in bma.match.group(0):
        # We found a subparen, increase the depth by 1
        new_depth = depth + 1
    elif right_bracket in bma.match.group(0):
        # We closed a paren, decrease the depth by 1
        new_depth = depth - 1

    # If we failed to finish, fold the match into the before text
    before_text += bma.middle

    # Recurse further
    return expand_to_target(
        bma.after,
        target=target,
        depth=new_depth,
        left_bracket=left_bracket,
        right_bracket=right_bracket,
        before_text=before_text,
    )
