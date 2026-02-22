import py_tools.utilities as ut

TSTRUT = "\\rule{0pt}{2.6ex}"
BSTRUT = "\\rule[-0.9ex]{0pt}{0pt}"


class Table:
    """LaTeX table builder.

    Stores table contents and formatting metadata and renders them as LaTeX
    ``tabular`` or ``tabu`` environments (optionally wrapped in a ``table``
    float).

    Parameters
    ----------
    contents : list of list, optional
        Row data.  Each sub-list represents one row; cells may be strings,
        floats, or any object that can be formatted with ``str()``.
        Defaults to an empty list.
    n_cols : int, optional
        Number of columns.  Inferred from the last row of ``contents`` when
        not provided.
    has_header : bool, optional
        If ``True`` the first row is treated as a header and a horizontal
        rule is automatically inserted after it.  Defaults to ``False``.
    alignment : str, optional
        LaTeX column alignment string (e.g. ``'lcc'``).  Defaults to
        ``'c'`` repeated for each column (or ``'X[c]'`` for ``tabu``).
    clines : dict, optional
        Partial horizontal rules keyed by row index.  Each value is a list
        of ``(start_col, end_col)`` tuples.  Defaults to ``{}``.
    hlines : list of int, optional
        Row indices after which a full horizontal rule is inserted.
        Defaults to ``[0]`` when ``has_header`` is ``True``, otherwise ``[]``.
    super_header : list of str, optional
        Text placed above the top rule spanning all columns.
    floatfmt : str, optional
        Python format string for floating-point cell values.  Defaults to
        ``'4.3f'``.
    tabu : bool, optional
        If ``True``, use the ``tabu`` package environment instead of the
        standard ``tabular``.  Defaults to ``False``.

    Attributes
    ----------
    contents : list of list
    n_cols : int
    has_header : bool
    alignment : str
    clines : dict
    hlines : list of int
    super_header : list of str or None
    floatfmt : str
    tabu : bool
    """

    def __init__(
        self,
        contents=None,
        n_cols=None,
        has_header=False,
        alignment=None,
        clines=None,
        hlines=None,
        super_header=None,
        floatfmt="4.3f",
        tabu=False,
    ):
        """Initialise the Table.

        Parameters
        ----------
        contents : list of list, optional
            Initial row data.  Defaults to an empty list.
        n_cols : int, optional
            Number of columns; inferred from ``contents`` when not supplied.
        has_header : bool, optional
            Whether the first row is a header.  Defaults to ``False``.
        alignment : str, optional
            LaTeX column alignment string.  Auto-generated when not supplied.
        clines : dict, optional
            Partial horizontal rules; see class docstring.  Defaults to ``{}``.
        hlines : list of int, optional
            Row indices for full horizontal rules.  Defaults to ``[0]`` when
            ``has_header`` is ``True``, otherwise ``[]``.
        super_header : list of str, optional
            Text placed above the top rule.
        floatfmt : str, optional
            Format string for float cells.  Defaults to ``'4.3f'``.
        tabu : bool, optional
            Use ``tabu`` environment.  Defaults to ``False``.
        """

        self.contents = contents if contents is not None else []
        self.n_cols = n_cols
        self.has_header = has_header
        self.alignment = alignment
        self.clines = clines if clines is not None else {}
        self.hlines = hlines
        self.super_header = super_header
        self.floatfmt = floatfmt
        self.tabu = tabu

        if self.hlines is None:
            if has_header:
                self.hlines = [0]
            else:
                self.hlines = []

        if self.n_cols is None:
            if self.contents:
                self.n_cols = len(self.contents[-1])
            else:
                self.n_cols = 0

        # Set alignment
        if self.alignment is None:
            if self.tabu:
                self.alignment = self.n_cols * "X[c]"
            else:
                self.alignment = self.n_cols * "c"

        if len(self.alignment) == 1 and self.n_cols > 1:
            self.alignment *= self.n_cols

    def n_rows(self):
        """Return the number of rows in the table.

        Returns
        -------
        int
            Length of ``self.contents``.
        """
        return len(self.contents)

    # Important kwargs = booktabs
    def table(
        self,
        caption=None,
        notes=None,
        position="h!",
        fontsize="small",
        caption_above=True,
        swp=False,
        **kwargs,
    ):
        """Format the table contents as a complete LaTeX ``table`` float.

        Parameters
        ----------
        caption : str, optional
            Caption text.  If ``None`` (default) no ``\\caption`` command is
            emitted.
        notes : str, optional
            Footnote text placed below the table body.  Defaults to ``None``.
        position : str, optional
            LaTeX float position specifier.  Defaults to ``'h!'``.
        fontsize : str, optional
            LaTeX font-size command (e.g. ``'small'``, ``'footnotesize'``).
            Defaults to ``'small'``.
        caption_above : bool, optional
            If ``True`` (default) the caption is placed above the tabular
            body; otherwise it is placed below.
        swp : bool, optional
            If ``True``, emit Scientific WorkPlace macro wrapping.  Defaults
            to ``False``.
        **kwargs
            Additional keyword arguments forwarded to :meth:`tabular`.

        Returns
        -------
        str
            Complete LaTeX source for the ``table`` float.
        """

        table_text = ""

        if swp:
            table_text += (
                r"""
%TCIMACRO{\TeXButton{B}{\begin{table}["""
                + position
                + r"""] \centering}}%
%BeginExpansion
"""
            )

        table_text += r"\begin{table}"

        table_text += "[{}]".format(position)

        if swp:
            table_text += r" \centering" + "\n" + r"%EndExpansion" + "\n"
        else:
            table_text += "\n" + r"\begin{center}" + "\n"

        if not swp:
            table_text += "\\" + fontsize + "\n"

        if caption is not None and caption_above:
            table_text += r"\caption{" + caption + "}\n"

        table_text += self.tabular(**kwargs)

        if not swp:
            table_text += r"\end{center}" + "\n"

        if caption is not None and not caption_above:
            table_text += r"\caption{" + caption + "}\n"

        if notes is not None:
            table_text += r"\footnotesize{" + notes + "}\n"

        if swp:
            table_text += r"""%TCIMACRO{\TeXButton{E}{\end{table}}}%
%BeginExpansion
"""
        table_text += r"\end{table}" + "\n"

        if swp:
            table_text += r"""%EndExpansion
"""

        return table_text

    def tabular(self, booktabs=True, width=r"\textwidth", include_tstrut=True):
        """Format the table contents as a LaTeX ``tabular`` (or ``tabu``) environment.

        Parameters
        ----------
        booktabs : bool, optional
            If ``True`` (default), use ``\\toprule``, ``\\midrule``, and
            ``\\bottomrule`` from the ``booktabs`` package.  Otherwise use
            ``\\hline \\hline``.
        width : str, optional
            Width argument for ``tabu`` environments.  Ignored for standard
            ``tabular``.  Defaults to ``r'\\textwidth'``.
        include_tstrut : bool, optional
            If ``True`` (default), add vertical-spacing struts around rows.

        Returns
        -------
        str
            LaTeX source for the ``tabular``/``tabu`` environment.
        """

        table_text = ""
        if self.tabu:
            table_text += r"\begin{tabu} to " + width + r" {" + self.alignment + "}\n"
        else:
            table_text += r"\begin{tabular}{" + self.alignment + "}\n"

        if self.super_header is not None:
            table_text += "\t&".join(self.super_header) + "\\\\\n"

        # Top rule
        if booktabs:
            table_text += r"\toprule" + "\n"
        else:
            table_text += r"\hline \hline" + "\n"

        table_text += self.body(booktabs=booktabs, include_tstrut=include_tstrut)

        if booktabs:
            table_text += r"\bottomrule" + "\n"
        else:
            table_text += r"\hline \hline" + "\n"

        if self.tabu:
            table_text += r"\end{tabu}" + "\n"
        else:
            table_text += r"\end{tabular}" + "\n"

        return table_text

    def body(self, booktabs=True, include_tstrut=True):
        """Render the table body rows as LaTeX source.

        Parameters
        ----------
        booktabs : bool, optional
            If ``True`` (default), use ``\\midrule`` for full horizontal
            rules; otherwise use ``\\hline``.
        include_tstrut : bool, optional
            If ``True`` (default), prepend a vertical strut to the first row
            and to each row that immediately follows a horizontal rule.

        Returns
        -------
        str
            LaTeX source for all body rows including inter-row rules.
        """
        table_text = ""

        # Body of table
        for i_row, row in enumerate(self.contents):
            for ii, entry in enumerate(row):
                if ii > 0:
                    table_text += "\t& "

                if isinstance(entry, float):
                    table_text += "{0:{1}}".format(entry, self.floatfmt)
                else:
                    table_text += "{}".format(entry)

            # Top strut (added to first row and each row after a rule)
            if include_tstrut:
                table_text += TSTRUT
                include_tstrut = False

            if (
                i_row in self.clines
                or i_row in self.hlines
                or i_row + 1 == self.n_rows()
            ):
                table_text += BSTRUT

            table_text += "\t" + r"\\"
            table_text += " \n"

            if i_row in self.clines:
                for start, end in self.clines[i_row]:
                    table_text += "\\cline{{{0}-{1}}}".format(start, end)
                table_text += "\n"
                include_tstrut = True
            elif i_row in self.hlines:
                if booktabs:
                    table_text += r"\midrule" + "\n"
                else:
                    table_text += r"\hline" + "\n"
                include_tstrut = True

        return table_text

    def row(self, i_row):
        """Return the contents of a single row.

        Parameters
        ----------
        i_row : int
            Zero-based row index.

        Returns
        -------
        list
            Cell values for row ``i_row``, or an empty list when ``i_row``
            is out of range.
        """

        if i_row < self.n_rows():
            return self.contents[i_row]
        else:
            return self.n_cols * []

    def add_cline(self, start, end, row=0):
        """Add a partial horizontal line below a specific row.

        Parameters
        ----------
        start : int
            First column of the partial rule (1-based, as in LaTeX
            ``\\cline``).
        end : int
            Last column of the partial rule (1-based).
        row : int, optional
            Zero-based row index below which the rule is placed.  Defaults to
            ``0``.

        Returns
        -------
        None
        """

        if row in self.clines:
            self.clines[row] += [(start, end)]
        else:
            self.clines[row] = [(start, end)]

        return None

    def multicolumn_row(self, text):
        """Create a single-element row containing a multicolumn cell spanning all columns.

        Parameters
        ----------
        text : str
            Content for the multicolumn cell.

        Returns
        -------
        list of str
            A one-element list containing the LaTeX ``\\multicolumn`` command
            string.
        """
        return [multicolumn(self.n_cols, text)]

    def add_super_header(self, text):
        """Set the super-header to a multicolumn cell spanning all columns.

        Parameters
        ----------
        text : str
            Content for the super-header cell.

        Returns
        -------
        None
        """
        self.super_header = self.multicolumn_row(text)


def multicolumn(n_cols, text):
    """Generate a LaTeX ``\\multicolumn`` command string.

    Parameters
    ----------
    n_cols : int
        Number of columns to span.
    text : str
        Content of the multicolumn cell.

    Returns
    -------
    str
        LaTeX ``\\multicolumn{n_cols}{c}{text}`` string.
    """
    return "\\multicolumn{{{0}}}{{c}}{{{1}}}".format(n_cols, text)


def hstack(table_list):
    """Stack a list of tables horizontally into a single table.

    Parameters
    ----------
    table_list : list of Table
        Tables to concatenate side by side.  They do not need to have the
        same number of rows; shorter tables are padded with empty rows.

    Returns
    -------
    Table
        New table whose column count equals the sum of the input column
        counts and whose rows are the element-wise concatenation of input
        rows.
    """
    n_rows = max([table.n_rows() for table in table_list])

    contents = [
        ut.join_lists([table.row(ii) for table in table_list]) for ii in range(n_rows)
    ]
    n_cols = sum([table.n_cols for table in table_list])
    alignment = "".join([table.alignment for table in table_list])
    hlines = list(set(ut.join_lists([table.hlines for table in table_list])))

    tabu = any([table.tabu for table in table_list])

    new_table = Table(
        contents, n_cols=n_cols, alignment=alignment, tabu=tabu, hlines=hlines
    )

    offset = 0
    for table in table_list:
        for row, start_end_list in table.clines.items():
            for start, end in start_end_list:
                new_table.add_cline(start + offset, end + offset, row)
        offset += table.n_cols

    return new_table


def empty_table(n_rows, n_cols, alignment=None, **kwargs):
    """Create a blank table filled with single-space strings.

    Parameters
    ----------
    n_rows : int
        Number of rows.
    n_cols : int
        Number of columns.
    alignment : str, optional
        LaTeX column alignment string.  Defaults to ``'c'`` repeated
        ``n_cols`` times.
    **kwargs
        Additional keyword arguments forwarded to the :class:`Table`
        constructor.

    Returns
    -------
    Table
        Empty table with ``n_rows`` rows and ``n_cols`` columns.
    """
    if alignment is None:
        alignment = n_cols * "c"

    contents = [n_cols * [" "] for row in range(n_rows)]
    return Table(contents, n_cols=n_cols, alignment=alignment, **kwargs)


def shift_down(table, new_row, is_header=True):
    """Prepend a new row to a table, shifting all existing rows down by one.

    Row indices in ``clines`` and ``hlines`` are incremented accordingly.

    Parameters
    ----------
    table : Table
        The table to modify (mutated in place and returned).
    new_row : list
        Cell values for the new first row.
    is_header : bool, optional
        Reserved for future use; currently unused.  Defaults to ``True``.

    Returns
    -------
    Table
        The mutated ``table`` with the new row prepended.
    """
    new_table = table
    new_table.contents = [new_row] + new_table.contents

    new_table.clines = {key + 1: val for key, val in table.clines.items()}

    new_table.hlines = [row + 1 for row in table.hlines]

    return new_table


# Join horizontally
def join_horizontal(table_list, header_list):
    """Join tables side by side, each with its own column header.

    Parameters
    ----------
    table_list : list of Table
        Tables to place next to each other.
    header_list : list of str or None
        Header text for each table.  A ``None`` entry means no header is
        added for the corresponding table.

    Returns
    -------
    Table
        Combined table with per-section column headers and a thin spacer
        column between sections that have multi-column headers.
    """
    tables_with_headers = []
    for table, header in zip(table_list, header_list):
        new_table = table

        add_header = header is not None

        if add_header:
            new_header = new_table.multicolumn_row(header)
        else:
            new_header = [" "]

        new_table = shift_down(new_table, new_header)

        if add_header:
            new_table.add_cline(1, new_table.n_cols)

        tables_with_headers.append(new_table)

    n_rows = max([table.n_rows() for table in tables_with_headers])

    full_table_list = []
    for i_table, (table, header) in enumerate(zip(tables_with_headers, header_list)):
        full_table_list.append(table)

        added_header = header is not None and table.n_cols > 1

        if i_table < len(tables_with_headers) - 1 and added_header:
            full_table_list.append(empty_table(n_rows, n_cols=1))

    super_table = hstack(full_table_list)

    return super_table


# Append vertically
def join_vertical(table_list, header_list):
    """Stack tables vertically with a section header above each one.

    All tables in ``table_list`` must have the same number of columns,
    the same alignment string, and the same ``tabu`` setting.

    Parameters
    ----------
    table_list : list of Table
        Tables to stack vertically; must share identical ``n_cols``,
        ``alignment``, and ``tabu`` attributes.
    header_list : list of str
        Section header text placed above each table.

    Returns
    -------
    Table
        Combined table with section headers and merged rules.

    Raises
    ------
    AssertionError
        If any table differs in ``n_cols``, ``alignment``, or ``tabu``.
    """
    assert all([table.n_cols == table_list[0].n_cols for table in table_list])
    assert all([table.alignment == table_list[0].alignment for table in table_list])
    assert all([table.tabu == table_list[0].tabu for table in table_list])

    n_cols = table_list[0].n_cols
    alignment = table_list[0].alignment
    tabu = table_list[0].tabu

    full_contents = []
    full_clines = {}
    full_hlines = []

    offset = 0
    for table, header in zip(table_list, header_list):
        # Add header
        if offset > 0:
            full_hlines.append(offset - 1)
        full_contents.append(table.multicolumn_row(header))
        full_hlines.append(offset)
        offset += 1

        # Add content
        full_contents += table.contents

        full_clines.update(
            {
                key + offset: [(start, end) for start, end in vals]
                for key, vals in table.clines.items()
            }
        )

        full_hlines += [val + offset for val in table.hlines]
        offset += table.n_rows()

    return Table(
        full_contents,
        n_cols,
        alignment=alignment,
        tabu=tabu,
        clines=full_clines,
        hlines=full_hlines,
    )


def open_latex(fid):
    """Write a standard LaTeX document preamble to a file object.

    Writes a ``\\documentclass`` declaration followed by commonly used
    packages, document-layout settings, and ``\\begin{document}`` with
    a few initial formatting commands.

    Parameters
    ----------
    fid : file-like object
        An open, writable file object.

    Returns
    -------
    None
    """
    fid.write(r"""
\documentclass[a4paper,12pt]{article}

\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{amsthm}
\usepackage{amsfonts}
\usepackage{changepage}
\usepackage{booktabs}
\usepackage{caption}
\usepackage{setspace}
\usepackage{tabu}

\allowdisplaybreaks[1]

\linespread{1.6}
%\linespread{1.3}

\begin{document}

\small

\setlength\voffset{-0.75 in}

\changepage{1.5 in}{1 in}{0 in}{-0.5 in}{0 in}{0 in}{0 in}{0 in}{0 in}
""")

    return None


def close_latex(fid):
    """Write the ``\\end{document}`` closing tag to a file object.

    Parameters
    ----------
    fid : file-like object
        An open, writable file object.

    Returns
    -------
    None
    """
    fid.write("\n" + r"\end{document}")
    return None


def regression_table(
    results,
    var_names=None,
    vertical=False,
    tstat=False,
    cov_type="HC0_se",
    floatfmt="4.3f",
    print_vars=None,
    stats=None,
    **kwargs,
):
    """Build a :class:`Table` from a regression results object.

    Parameters
    ----------
    results : statsmodels results object
        A fitted regression results instance exposing ``.params``,
        ``.tvalues``, and optionally ``.rsquared`` / ``.rsquared_adj``.
    var_names : list of str, optional
        Variable names to use as column headers (horizontal layout) or row
        labels.  Required when ``print_vars`` is specified.
    vertical : bool, optional
        If ``True``, format the table in a single-column (vertical) layout
        with each coefficient on its own row.  Defaults to ``False``.
    tstat : bool, optional
        If ``True``, report t-statistics instead of standard errors in
        parentheses.  Defaults to ``False``.
    cov_type : str, optional
        Attribute name of the results object from which standard errors are
        read (e.g. ``'HC0_se'``).  Ignored when ``tstat`` is ``True``.
        Defaults to ``'HC0_se'``.
    floatfmt : str, optional
        Python format string for floating-point values.  Defaults to
        ``'4.3f'``.
    print_vars : list of str, optional
        Subset of ``var_names`` to include.  All variables are included when
        ``None`` (default).
    stats : list of str, optional
        Names of scalar attributes on ``results`` to append as summary
        statistics.  Defaults to ``['rsquared_adj']``.
    **kwargs
        Additional keyword arguments forwarded to the :class:`Table`
        constructor.

    Returns
    -------
    Table
        Formatted regression table.
    """
    if stats is None:
        stats = ["rsquared_adj"]

    tex_stats = {"rsquared": "R^2", "rsquared_adj": "\\bar{R}^2"}

    contents = []

    coeffs = results.params

    if tstat:
        se_like = results.tvalues
    else:
        se_like = getattr(results, cov_type)

    has_header = False

    n_vars = len(coeffs)
    if print_vars is not None:
        print_ix = [ii for ii in range(n_vars) if var_names[ii] in print_vars]
    else:
        print_vars = var_names
        print_ix = range(n_vars)

    if vertical:
        for ii in range(n_vars):
            contents.append([coeffs[ii]])
            contents.append(["({0:{1}})".format(se_like[ii], floatfmt)])
        for stat in stats:
            this_stat = getattr(results, stat)
            contents.append(["{0:{1}}".format(this_stat, floatfmt)])
    else:
        if var_names is not None:
            header_list = print_vars + [tex_stats[stat] for stat in stats]
            contents.append(header_list)
            has_header = True

        contents.append(
            [coeffs[ii] for ii in print_ix] + [getattr(results, stat) for stat in stats]
        )

        contents.append(
            ["({0:{1}})".format(se_like[ii], floatfmt) for ii in print_ix]
            + [" " for stat in stats]
        )

    return Table(contents, has_header=has_header, floatfmt=floatfmt, **kwargs)


def to_camel_case(s: str) -> str:
    """Convert a ``snake_case`` string to ``camelCase``.

    Parameters
    ----------
    s : str
        Input string in ``snake_case`` format.

    Returns
    -------
    str
        The input string converted to ``camelCase``.

    Examples
    --------
    >>> to_camel_case('my_variable_name')
    'myVariableName'
    """
    parts = s.split("_")
    return parts[0] + "".join(word.capitalize() for word in parts[1:])


def write_values_tex(
    data: dict,
    path="values.tex",
    prop_name="g_myvals_prop",
    prefix=None,
    command_str=None,
    add_prefix_to_command=True,
):
    """Write a LaTeX file defining key-value pairs accessible as ``\\val{key}``.

    Uses the LaTeX3 ``expl3`` property-list interface.  The generated file
    can be ``\\input``-ed into a LaTeX document to expose the data values as
    a single accessor command.

    Parameters
    ----------
    data : dict
        Mapping of string keys to values.  Values are converted to strings
        with ``str()``; brace characters are escaped.
    path : str, optional
        Output file path.  Defaults to ``'values.tex'``.
    prop_name : str, optional
        Name of the LaTeX3 property-list variable (without the leading
        backslash).  Defaults to ``'g_myvals_prop'``.
    prefix : str, optional
        If provided, all keys in the output are prefixed with
        ``'<prefix>_'``.  When ``add_prefix_to_command`` is also ``True``
        and ``prefix`` is not ``None``, the accessor command name is derived
        from ``'val_<prefix>'`` converted to camelCase.
    command_str : str, optional
        Name of the LaTeX accessor command (without the leading backslash).
        Overrides the auto-generated name from ``prefix``.  Defaults to
        ``'val'`` when neither ``prefix`` nor ``command_str`` is specified.
    add_prefix_to_command : bool, optional
        If ``True`` (default) and ``prefix`` is set, the accessor command
        name includes the prefix converted to camelCase.

    Returns
    -------
    None

    Examples
    --------
    Example output for ``data={'alpha': 0.05}`` with no prefix (default
    accessor command ``\\val``)::

        \\ExplSyntaxOn
        \\prop_new:N \\g_myvals_prop
        \\prop_gput:Nnn \\g_myvals_prop {alpha} {0.05}
        \\cs_new:Npn \\val #1 { \\prop_item:Nn \\g_myvals_prop {#1} }
        \\ExplSyntaxOff

    With ``prefix='params'`` and ``add_prefix_to_command=True`` the accessor
    command becomes ``\\valParams`` and each key is prefixed::

        \\prop_gput:Nnn \\g_myvals_prop {params_alpha} {0.05}
        \\cs_new:Npn \\valParams #1 { \\prop_item:Nn \\g_myvals_prop {#1} }
    """
    if command_str is None:
        if add_prefix_to_command and (prefix is not None):
            command_str = to_camel_case("val_" + prefix)
        else:
            command_str = "val"

    with open(path, "w", encoding="utf-8") as f:
        f.write("\\ExplSyntaxOn\n")
        f.write(f"\\prop_new:N \\{prop_name}\n")

        for key, value in data.items():
            if prefix is None:
                key_new = key
            else:
                key_new = f"{prefix}_{key}"
            # Convert Python values to TeX-safe strings
            val_str = str(value)
            # Escape braces if they appear in value strings
            val_str = val_str.replace("{", "\\{").replace("}", "\\}")
            f.write(f"\\prop_gput:Nnn \\{prop_name} {{{key_new}}} {{{val_str}}}\n")

        f.write(
            f"\\cs_new:Npn \\{command_str} #1 {{ \\prop_item:Nn \\{prop_name} {{#1}} }}\n"
        )
        f.write("\\ExplSyntaxOff\n")
