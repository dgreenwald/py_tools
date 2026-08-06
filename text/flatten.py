#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 14:34:39 2025

@author: dan
"""

import os
import shutil
from py_tools.text import parsing as par
import re
from collections import namedtuple

Command = namedtuple("Command", ["name", "nargs", "content"])


def get_next_argument(text):
    """Extract the next ``{...}`` argument from the beginning of a text string.

    Parameters
    ----------
    text : str
        Input text.  The next ``{...}`` block (possibly preceded by
        whitespace) is extracted from the front.

    Returns
    -------
    argument : str
        Content of the ``{...}`` block (without the surrounding braces).
    after : str
        Remainder of ``text`` following the closing brace.

    Raises
    ------
    Exception
        If the text does not start with optional whitespace followed by
        ``'{'``.
    """
    match = re.match(r"\s*{", text)
    bma_init = par.match_to_bma(match, text)
    if not bma_init.matched:
        raise Exception

    argument, _, after = par.expand_to_target(bma_init.after, left_bracket="{")

    return argument, after


def get_arguments(text, nargs):
    """Extract multiple consecutive ``{...}`` arguments from a text string.

    Parameters
    ----------
    text : str
        Input text from which ``nargs`` brace-delimited arguments are
        extracted sequentially.
    nargs : int
        Number of arguments to extract.

    Returns
    -------
    arguments : list of str
        Extracted argument contents (without surrounding braces), in order.
    text : str
        Remainder of ``text`` after all extracted arguments.
    """
    arguments = []
    for ii in range(nargs):
        argument, text = get_next_argument(text)
        arguments.append(argument)

    return arguments, text


def replace_content(text, command):
    """Substitute the next ``command.nargs`` arguments into a command's body.

    Reads exactly ``command.nargs`` ``{...}`` arguments from the front of
    ``text``, substitutes them into ``command.content`` (replacing ``#1``,
    ``#2``, … placeholders), and returns the expanded content together with
    the remaining text.

    Parameters
    ----------
    text : str
        Input text positioned immediately after the command name (i.e. the
        arguments follow at the start of the string).
    command : Command
        Named tuple with fields ``name``, ``nargs``, and ``content``.

    Returns
    -------
    new_content : str
        The command body with all argument placeholders substituted.
    text : str
        Remainder of ``text`` after all consumed arguments.

    Raises
    ------
    ValueError
        If ``command.nargs >= 10`` (double-digit argument numbers are not
        supported) or if the number of extracted arguments does not match
        ``command.nargs``.
    """
    # If we go to double digits there will be issues with the regex, where #10 catches #1, etc.
    if command.nargs >= 10:
        raise ValueError(
            "replace_content only supports commands with fewer than 10 arguments."
        )

    new_content = command.content
    arguments, text = get_arguments(text, command.nargs)
    if len(arguments) != command.nargs:
        raise ValueError(
            "Unexpected argument count while replacing LaTeX command content."
        )
    for ii, argument in enumerate(arguments):
        arg_no = ii + 1
        new_content = re.sub(rf"#{arg_no}", re.escape(argument), new_content)

    return new_content, text


def replace_command(text, command):
    """Replace all occurrences of a single LaTeX command in ``text``.

    Finds the first occurrence of ``\\<command.name>`` followed by a
    non-word character, substitutes the next ``command.nargs`` brace
    arguments, and then recurses on the remainder until no more occurrences
    are found.

    Parameters
    ----------
    text : str
        LaTeX source text to process.
    command : Command
        Named tuple with fields ``name``, ``nargs``, and ``content``.

    Returns
    -------
    replaced_text : str
        Text with all occurrences of the command replaced.
    replaced_any : bool
        ``True`` if at least one replacement was made.
    """
    pattern = r"\\" + command.name + r"(?=[\W_])"
    bma = par.bma_search(pattern, text)
    if not bma.matched:
        replaced_any = False
        return text, replaced_any

    # Substitute out the command
    new_content, after_text = replace_content(bma.after, command)

    # Add remaining text
    after_new, _ = replace_command(after_text, command)

    replaced_text = bma.before + new_content + after_new
    replaced_any = True
    return replaced_text, replaced_any


def replace_commands_static(text, commands):
    """Replace all occurrences of a list of LaTeX commands in ``text``.

    Repeatedly iterates over ``commands`` calling :func:`replace_command`
    until a full pass results in no replacements (i.e. all command
    occurrences have been expanded).

    Parameters
    ----------
    text : str
        LaTeX source text to process.
    commands : list of Command
        Commands to replace; each is a named tuple with ``name``, ``nargs``,
        and ``content`` fields.

    Returns
    -------
    str
        Text with all command occurrences fully expanded.
    """
    # Loop through until we stop finding any commands
    done = False
    while not done:
        done = True

        # Loop through commands
        for command in commands:
            text, replaced_any = replace_command(text, command)
            done = done and (not replaced_any)

    return text


def remove_comments(text, replaced_text=""):
    """Strip LaTeX comments from ``text``.

    Removes everything from an unescaped ``%`` character to the end of its
    line (inclusive).  The function is called recursively until all comment
    regions are removed.

    Parameters
    ----------
    text : str
        LaTeX source text that may contain ``%`` comments.
    replaced_text : str, optional
        Accumulated output from previous recursive calls.  Defaults to
        ``''``.

    Returns
    -------
    str
        Text with all comment regions removed.
    """
    pattern = r"(?<!\\)%"
    bma = par.bma_search(pattern, text)
    if not bma.matched:
        return text

    # Add
    replaced_text += bma.before
    eol_pattern = r"\n"
    bma_eol = par.bma_search(eol_pattern, bma.after)
    if bma_eol.matched:
        # We found the end of the line, proceed from there
        replaced_text += remove_comments(bma_eol.after)

    return replaced_text


def get_definitions(text):
    """Extract all ``\\def`` command definitions from LaTeX source text.

    Parameters
    ----------
    text : str
        LaTeX source text (typically the preamble) to search for ``\\def``
        definitions.

    Returns
    -------
    list of Command
        A list of :class:`Command` named tuples (``name``, ``nargs=0``,
        ``content``) for each ``\\def`` found.
    """
    defn_pattern = r"^\s*\\def\s*\\(\w*)\s*{\s*"

    defn_list = []
    for match in re.finditer(defn_pattern, text, re.MULTILINE):
        bma = par.match_to_bma(match, text)
        name = bma.match.groups(1)[0]
        content, _, _ = par.expand_to_target(bma.after, left_bracket="{")

        defn_list.append(Command(name, 0, content))

    return defn_list


def remove_command(text, command):
    """Remove a ``\\newcommand`` declaration from LaTeX source text.

    Constructs a regex that matches the full ``\\newcommand{\\<name>}[nargs]{content}``
    declaration line and removes it from ``text``.

    Parameters
    ----------
    text : str
        LaTeX source text.
    command : Command
        Named tuple with ``name``, ``nargs``, and ``content`` fields
        describing the command declaration to remove.

    Returns
    -------
    str
        Text with the matching ``\\newcommand`` declaration removed.
    """
    # Name of command
    pattern = r"\\newcommand\s*{\s*\\" + re.escape(command.name) + r"}"

    # Number of arguments
    if command.nargs > 0:
        pattern += rf"\s*\[{command.nargs:d}\]"

    # Content
    pattern += r"\s*{" + re.escape(command.content) + r"\s*}\s*\n"

    # Replace
    text = re.sub(pattern, "", text)

    return text


def remove_commands(text, command_list):
    """Remove multiple ``\\newcommand`` definitions from LaTeX source text.

    Parameters
    ----------
    text : str
        LaTeX source text.
    command_list : list of Command
        Commands to remove; each is a named tuple with ``name``, ``nargs``,
        and ``content`` fields.

    Returns
    -------
    str
        Text with all matching ``\\newcommand`` declarations removed.
    """
    for command in command_list:
        text = remove_command(text, command)

    return text


def remove_defn(text, command):
    """Remove a ``\\def`` definition from LaTeX source text.

    Parameters
    ----------
    text : str
        LaTeX source text.
    command : Command
        Named tuple with ``name`` and ``content`` fields describing the
        ``\\def`` command to remove.

    Returns
    -------
    str
        Text with the matching ``\\def`` declaration removed.
    """
    # Name of command
    pattern = r"\\def\s*\\" + re.escape(command.name)

    # Content
    pattern += r"\s*{" + re.escape(command.content) + r"\s*}\s*"

    # Replace
    text = re.sub(pattern, "", text)

    return text


def remove_defns(text, defn_list):
    """Remove multiple ``\\def`` definitions from LaTeX source text.

    Parameters
    ----------
    text : str
        LaTeX source text.
    defn_list : list of Command
        ``\\def`` commands to remove.

    Returns
    -------
    str
        Text with all matching ``\\def`` declarations removed.
    """
    for defn in defn_list:
        text = remove_defn(text, defn)

    return text


def replace_ref(text, label, replacement):
    """Replace a single LaTeX ``\\ref`` or ``\\eqref`` with a literal string.

    Parameters
    ----------
    text : str
        LaTeX source text.
    label : str
        The label argument of ``\\ref{label}`` or ``\\eqref{label}`` to
        replace.
    replacement : str
        Literal text to substitute in place of the reference command.  For
        ``\\eqref`` the replacement is additionally wrapped in parentheses.

    Returns
    -------
    str
        Text with all matching reference commands replaced.
    """
    pattern = r"\\ref\s*{\s*" + label + r"}"
    text = re.sub(pattern, replacement, text)

    pattern = r"\\eqref\s*{\s*" + label + r"}"
    text = re.sub(pattern, "(" + replacement + ")", text)
    return text


def replace_refs(text, refs_to_replace):
    """Replace multiple ``\\ref`` / ``\\eqref`` labels with literal text.

    Parameters
    ----------
    text : str
        LaTeX source text.
    refs_to_replace : list of tuple
        Each entry is a ``(label, replacement)`` pair forwarded to
        :func:`replace_ref`.

    Returns
    -------
    str
        Text with all specified reference commands replaced.
    """
    for label, replacement in refs_to_replace:
        text = replace_ref(text, label, replacement)

    return text


def get_commands(text):
    """Extract all ``\\newcommand`` definitions from LaTeX source text.

    Parameters
    ----------
    text : str
        LaTeX source text (typically the preamble) to search for
        ``\\newcommand`` definitions.

    Returns
    -------
    list of Command
        A list of :class:`Command` named tuples (``name``, ``nargs``,
        ``content``) for each ``\\newcommand`` found.
    """
    command_pattern = r"^\s*\\newcommand{\s*\\"
    nargs_pattern = r"\[\s*(\d*)\s*\]"
    start_bracket_pattern = r"\s*{"

    command_list = []
    for match in re.finditer(command_pattern, text, re.MULTILINE):
        bma = par.match_to_bma(match, text)
        name, _, after = par.expand_to_target(bma.after, left_bracket="{")

        # Get number of arguments
        match_nargs = re.match(nargs_pattern, after)
        if match_nargs is None:
            # No arguments
            nargs = 0
        else:
            # Has argument number
            nargs = match_nargs.groups(1)[0]
            if nargs == "":
                nargs = 0
            else:
                nargs = int(nargs)

            # Skip past this
            after = after[match_nargs.end() :]

        match_start_bracket = re.match(start_bracket_pattern, after)
        assert match_start_bracket is not None
        after = after[match_start_bracket.end() :]
        content, _, _ = par.expand_to_target(after, left_bracket="{")

        command_list.append(Command(name, nargs, content))

    return command_list


def read_if_exists(filepath):
    """Read the contents of a file if it exists, otherwise return ``None``.

    Parameters
    ----------
    filepath : str
        Path to the file to read.

    Returns
    -------
    str or None
        File contents as a string, or ``None`` if the file does not exist.
    """
    if os.path.exists(filepath):
        with open(filepath, "rt") as fid:
            return fid.read()
    else:
        return None


def _read_input_file_and_path(filepath, base_dir=None):
    """Read a LaTeX ``\\input`` target and return its text plus resolved path."""
    filepath = remove_brackets(filepath)

    candidates = [filepath, filepath + ".tex"]
    if base_dir is not None:
        candidates.extend(
            [
                os.path.join(base_dir, filepath),
                os.path.join(base_dir, filepath + ".tex"),
            ]
        )

    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        text = read_if_exists(candidate)
        if text is not None:
            return text, candidate

    return None, None


def remove_brackets(text):
    """Recursively strip all outermost ``{...}`` braces from a text string.

    Parameters
    ----------
    text : str
        Input text potentially containing ``{...}`` groups.

    Returns
    -------
    str
        Text with the contents of each ``{...}`` group unwrapped (i.e. the
        braces removed and the inner text preserved).
    """
    bma = par.bma_search(r"{", text)
    if bma.matched:
        argument, _, after = par.expand_to_target(bma.after, left_bracket="{")
        argument_new = remove_brackets(argument)
        after_new = remove_brackets(after)
        return bma.before + argument_new + after_new
    else:
        return text


def read_input_file(filepath, base_dir=None):
    """Read a LaTeX ``\\input`` target file, trying with and without ``.tex``.

    Parameters
    ----------
    filepath : str
        File path as it appears in the ``\\input{...}`` argument.  Leading/
        trailing whitespace should already be stripped; any surrounding
        braces are removed via :func:`remove_brackets`.

    Returns
    -------
    str or None
        File contents if the file (or the file with ``.tex`` appended) is
        found, otherwise ``None``.
    """
    text, _ = _read_input_file_and_path(filepath, base_dir=base_dir)
    return text


def remove_unused_commands(text, command_list=None, defn_list=None):
    """Remove command/definition declarations that appear only once in ``text``.

    A declaration is considered "unused" if the command name appears only
    once in the text (i.e. only in the declaration itself, not in any usage
    site).  The removal loop repeats until no more declarations can be
    eliminated.

    Parameters
    ----------
    text : str
        LaTeX source text.
    command_list : list of Command, optional
        ``\\newcommand`` declarations to check.  Defaults to ``[]``.
    defn_list : list of Command, optional
        ``\\def`` declarations to check.  Defaults to ``[]``.

    Returns
    -------
    str
        Text with all unused command/definition declarations removed.
    """
    if command_list is None:
        command_list = []
    if defn_list is None:
        defn_list = []

    done = False
    while not done:
        done = True
        for command in command_list:
            matches = re.findall(r"\\" + command.name + r"\b", text)
            if len(matches) == 1:
                done = False
                text = remove_command(text, command)

        for defn in defn_list:
            matches = re.findall(r"\\" + defn.name + r"\b", text)
            if len(matches) == 1:
                done = False
                text = remove_defn(text, defn)

    return text


def replace_commands_dynamic(text, commands_to_replace, names_to_replace=None):
    """Expand LaTeX commands in ``text``, updating definitions as they are encountered.

    Processes the text in segments separated by ``\\def`` declarations.
    Within each segment, commands in ``commands_to_replace`` are expanded
    statically.  When a ``\\def`` for a name listed in ``names_to_replace``
    is reached, the command list is updated before processing continues,
    allowing later occurrences to use the new definition.

    Parameters
    ----------
    text : str
        LaTeX source text to process.
    commands_to_replace : list of Command
        Current set of commands to expand.
    names_to_replace : list of str, optional
        Command names whose ``\\def`` re-declarations should be tracked and
        incorporated.  Defaults to ``[]``.

    Returns
    -------
    str
        Text with all tracked commands expanded.
    """
    current_names = [cmd.name for cmd in commands_to_replace]
    if names_to_replace is None:
        names_to_replace = []

    # Need to dynamically update the list of definitions, so only replace until next \def
    defn_pattern = r"^\s*\\def\s*\\(\w*)\s*{\s*"
    bma = par.bma_search(defn_pattern, text, re.MULTILINE)

    # Replace up to this point using current list
    before = replace_commands_static(bma.before, commands_to_replace)

    if bma.matched:
        # If we have a new definition, need to update the list
        commands_to_replace_new = commands_to_replace.copy()

        # Get the new definition name
        name = bma.match.groups(1)[0]

        # Expand through the argument
        content, bracket, after = par.expand_to_target(bma.after, left_bracket="{")

        # Only update if name is on the list
        if name in names_to_replace:
            # Get the new definition
            this_defn = Command(name, 0, content)

            # Check if it is already in the list
            if name in current_names:
                # This is replacing an existing definition
                ix = current_names.index(name)
                commands_to_replace_new[ix] = this_defn
            else:
                # Otherwise, append to the end of the list
                commands_to_replace_new.append(this_defn)

        else:
            before += bma.middle + content + bracket

        # Now replace the rest of the text
        after = replace_commands_dynamic(
            after, commands_to_replace_new, names_to_replace=names_to_replace
        )

    else:
        # Otherwise, we are done
        after = ""

    text_new = before + after

    return text_new


def get_aux_labels(aux_file):
    """Parse a LaTeX ``.aux`` file and extract label-to-number mappings.

    Parameters
    ----------
    aux_file : str
        Path to the ``.aux`` file generated by a LaTeX compilation.

    Returns
    -------
    list of tuple of (str, str)
        Each entry is a ``(label, doc_number)`` pair where ``label`` is the
        LaTeX cross-reference key and ``doc_number`` is the document-assigned
        reference number (e.g. ``'3.2'`` for equation 3.2).
    """
    with open(aux_file, "rt") as fid:
        aux_lines = fid.readlines()

    labels_to_numbers = []
    for line in aux_lines:
        pattern = r"\\newlabel{"
        bma = par.bma_search(pattern, line)
        if bma.matched:
            label, matched_bracket, after = par.expand_to_target(
                bma.after, left_bracket="{"
            )
            doc_number_match = re.match(r"{{(\w+\.?\w*)}", after)
            doc_number = doc_number_match.groups(1)[0]
            labels_to_numbers.append((label, doc_number))

    return labels_to_numbers


def get_figure_labels(text):
    """Extract ``(label, image_path)`` pairs from LaTeX ``figure`` environments.

    Parameters
    ----------
    text : str
        LaTeX source text containing ``\\begin{figure}...\\end{figure}``
        blocks.

    Returns
    -------
    list of tuple of (str, str)
        Each entry is a ``(label, image_path)`` pair extracted from a figure
        environment that contains both an ``\\includegraphics`` and a
        ``\\label`` command.
    """
    figure_blocks = re.findall(r"\\begin{figure}.*?\\end{figure}", text, re.DOTALL)

    figures = []
    for block in figure_blocks:
        image_match = re.search(r"\\includegraphics(?:\[.*?\])?{(.*?)}", block)
        label_match = re.search(r"\\label{(.*?)}", block)

        if image_match and label_match:
            figures.append((label_match.group(1), image_match.group(1)))

    return figures


def replace_figures(
    text,
    aux_file,
    figure_dir_in_text=None,
    figure_dir_actual_location=None,
    base_dir=None,
):
    """Rename and copy figure files, updating their paths in the LaTeX source.

    For each figure whose label appears in the ``.aux`` file, the image is
    copied to ``figure_dir_actual_location`` with a sequentially numbered
    filename (e.g. ``fig3.2.pdf``), and the ``\\includegraphics`` path in
    ``text`` is updated accordingly.

    Parameters
    ----------
    text : str
        LaTeX source text.
    aux_file : str
        Path to the ``.aux`` file used to map labels to document numbers.
    figure_dir_in_text : str, optional
        Directory prefix to use for figure paths in the output LaTeX text.
        Defaults to ``''`` (current directory).
    figure_dir_actual_location : str, optional
        Filesystem directory where renamed figure files are copied.
        Defaults to ``None``; callers must supply a valid directory.

    Returns
    -------
    str
        Updated LaTeX source text with replaced figure paths.
    """
    if figure_dir_in_text is None:
        figure_dir_in_text = ""

    figures = get_figure_labels(text)
    aux_labels = get_aux_labels(aux_file)
    aux_labels_dict = {label: doc_number for label, doc_number in aux_labels}

    for label, image_path in figures:
        stem, ext = os.path.splitext(image_path)
        if ext == "":
            ext = ".pdf"
            image_path = stem + ext

        image_path_fs = image_path
        if (base_dir is not None) and (not os.path.isabs(image_path_fs)):
            image_path_fs = os.path.join(base_dir, image_path_fs)

        if os.path.exists(image_path_fs) and (label in aux_labels_dict):
            # _, ext = os.path.splitext(image_path)
            doc_number = aux_labels_dict[label]
            new_path = os.path.join(figure_dir_actual_location, f"fig{doc_number}{ext}")
            new_path_in_text = os.path.join(figure_dir_in_text, f"fig{doc_number}{ext}")
            shutil.copy2(image_path_fs, new_path)

            text = re.sub(
                rf"(\\includegraphics(?:\[.*?\])?){{{re.escape(stem)}(?:{re.escape(ext)})?}}",
                rf"\1{{{new_path_in_text}}}",
                text,
            )
        else:
            print(f"Could not replace {image_path}")

    return text


def flatten_text(
    text,
    flattened_text="",
    commands_to_replace=None,
    do_remove_comments_from_text=True,
    names_to_replace=None,
    base_dir=None,
):
    """Recursively expand ``\\input{...}`` directives in a LaTeX text string.

    Processes ``text`` by replacing each ``\\input{filepath}`` with the
    contents of the referenced file (recursively).  Optionally strips
    comments and expands user-defined commands before processing inputs.

    Parameters
    ----------
    text : str
        LaTeX source text to flatten.
    flattened_text : str, optional
        Accumulated output from previous recursive calls.  Defaults to
        ``''``.
    commands_to_replace : list of Command, optional
        Commands to expand dynamically as defined in
        :func:`replace_commands_dynamic`.  Defaults to ``[]``.
    do_remove_comments_from_text : bool, optional
        If ``True`` (default), strip ``%`` comments before processing.
    names_to_replace : list of str, optional
        Command names whose ``\\def`` re-declarations should be tracked;
        forwarded to :func:`replace_commands_dynamic`.

    Returns
    -------
    str
        Fully flattened LaTeX source.
    """
    # Replace any definitions in the body of the text
    if commands_to_replace is None:
        commands_to_replace = []

    if do_remove_comments_from_text:
        text = remove_comments(text)

    text = replace_commands_dynamic(
        text, commands_to_replace, names_to_replace=names_to_replace
    )

    pattern = r"\\input\s*{"

    bma = par.bma_search(pattern, text)
    if not bma.matched:
        flattened_text += text
        return flattened_text

    # Add text up to input
    flattened_text += bma.before

    # Capture text inside brackets
    argument, matched_bracket, after_input = par.expand_to_target(
        bma.after, left_bracket="{"
    )

    # Try to read from file
    filepath = argument.strip()
    input_text, input_path = _read_input_file_and_path(filepath, base_dir=base_dir)

    if input_text is not None:
        # print(f'loaded {filepath}')
        # It worked, load in the text (after recursively flattening)
        flattened_text += flatten_text(
            input_text,
            commands_to_replace=commands_to_replace,
            names_to_replace=names_to_replace,
            base_dir=os.path.dirname(os.path.abspath(input_path)),
        )
    else:
        # It didn't work, leave it as-is
        print(f"failed to load {filepath}")
        flattened_text += bma.middle + argument + matched_bracket

    # Continue through rest of the file
    flattened_text += flatten_text(
        after_input,
        commands_to_replace=commands_to_replace,
        names_to_replace=names_to_replace,
        base_dir=base_dir,
    )

    return flattened_text


def flatten(
    infile=None,
    outfile=None,
    names_to_replace=None,
    aux_reference_file=None,
    do_remove_comments_from_text=True,
    do_remove_comments_from_preamble=False,
    do_remove_unused=False,
    do_replace_figures=False,
    figure_dir_in_text=None,
    figure_dir_actual_location=None,
):
    """Flatten a multi-file LaTeX project into a single self-contained source file.

    Reads ``infile``, recursively expands all ``\\input{...}`` directives,
    optionally strips comments, expands user-defined commands, replaces
    cross-references with literal numbers from an auxiliary file, removes
    unused commands, and copies/renames figure files.

    Parameters
    ----------
    infile : str
        Path to the top-level ``.tex`` source file.
    outfile : str or None
        Path to write the flattened output.  If ``None``, the flattened text
        is returned instead of being written to disk.
    names_to_replace : list of str, optional
        Command names whose ``\\newcommand``/``\\def`` definitions should be
        expanded in the document body.  Defaults to ``[]``.
    aux_reference_file : str, optional
        Path to a ``.aux`` file used to replace ``\\ref``/``\\eqref``
        commands with literal document numbers.  Defaults to ``None``
        (no reference replacement).
    do_remove_comments_from_text : bool, optional
        If ``True`` (default), strip ``%`` comments from the document body.
    do_remove_comments_from_preamble : bool, optional
        If ``True``, also strip comments from the preamble.  Defaults to
        ``False``.
    do_remove_unused : bool, optional
        If ``True``, remove ``\\newcommand``/``\\def`` declarations that are
        not used in the flattened document.  Defaults to ``False``.
    do_replace_figures : bool, optional
        If ``True``, rename figure files and update their paths; requires
        a ``.aux`` file adjacent to ``infile``.  Defaults to ``False``.
    figure_dir_in_text : str, optional
        Directory prefix used for figure paths in the output LaTeX text.
        Only relevant when ``do_replace_figures`` is ``True``.
    figure_dir_actual_location : str, optional
        Filesystem directory where renamed figure files are copied.
        Defaults to the directory of ``outfile`` when ``do_replace_figures``
        is ``True``.

    Returns
    -------
    str or None
        The flattened LaTeX source when ``outfile`` is ``None``; otherwise
        writes to ``outfile`` and returns ``None``.
    """
    if names_to_replace is None:
        names_to_replace = []

    infile_abs = os.path.abspath(infile)
    infile_dir = os.path.dirname(infile_abs)

    if outfile is None:
        outfile_path = None
        head_out = ""
    elif os.path.isabs(outfile):
        outfile_path = outfile
        head_out, _ = os.path.split(outfile_path)
    else:
        outfile_path = os.path.join(infile_dir, outfile)
        head_out, _ = os.path.split(outfile_path)

    with open(infile_abs, "rt") as fid:
        text = fid.read()

    bma_preamble = par.bma_search(r"\\begin\s*{\s*document\s*}", text)
    assert bma_preamble.matched
    # match_preamble = re.search(r'\\begin\s*{\s*document\s*}')

    # Get commands and definitions
    defn_list = get_definitions(bma_preamble.before)
    command_list = get_commands(bma_preamble.before)

    # Filter down to selected list
    commands_to_replace = [cmd for cmd in command_list if cmd.name in names_to_replace]
    defns_to_replace = [cmd for cmd in defn_list if cmd.name in names_to_replace]
    commands_and_defns_to_replace = commands_to_replace + defns_to_replace

    # Flatten the text after preamble
    after_preamble_new = flatten_text(
        bma_preamble.after,
        commands_to_replace=commands_and_defns_to_replace,
        do_remove_comments_from_text=do_remove_comments_from_text,
        names_to_replace=names_to_replace,
        base_dir=infile_dir,
    )

    # Remove preamble comments?
    preamble_new = bma_preamble.before
    if do_remove_comments_from_preamble:
        preamble_new = remove_comments(preamble_new)

    # Reassemble
    text = preamble_new + bma_preamble.middle + after_preamble_new

    # Replace references from auxiliary document
    if aux_reference_file is not None:
        if not os.path.isabs(aux_reference_file):
            aux_reference_file = os.path.join(infile_dir, aux_reference_file)
        refs_to_replace = get_aux_labels(aux_reference_file)
        text = replace_refs(text, refs_to_replace)

    # Get rid of unneeded commands
    if do_remove_unused:
        text = remove_unused_commands(
            text, command_list=command_list, defn_list=defn_list
        )

    # Replace figures
    if do_replace_figures:
        this_aux_file = infile_abs.replace(".tex", ".aux")
        if figure_dir_actual_location is None:
            figure_dir_actual_location = head_out + "/"
        text = replace_figures(
            text,
            this_aux_file,
            figure_dir_in_text=figure_dir_in_text,
            figure_dir_actual_location=figure_dir_actual_location,
            base_dir=infile_dir,
        )

    if outfile_path is not None:
        with open(outfile_path, "wt") as fid:
            fid.write(text)
        return None

    return text
