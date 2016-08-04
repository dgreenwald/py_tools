from tabulate import tabulate

import py_tools.utilities as ut

class Table:
    """Latex table class.
    
    contents: list of lists, each sublist is one row
    n_cols: number of columns of the table
    has_header: flag for whether the top row is a header row
    alignment: alignment of columns (e.g., 'lcc') 
    clines: additional partial underlines
    hlines: additional full underlines
    super_header: text above top line
    floatfmt: format for floating point entries
    """

    def __init__(self, contents=None, n_cols=None, has_header=False, 
                 alignment=None, clines=None, hlines=None,
                 super_header=None,
                 floatfmt='4.3f'):

        self.contents = contents if contents is not None else []
        self.n_cols = n_cols
        self.has_header = has_header
        self.alignment = alignment
        self.clines = clines if clines is not None else {}
        self.hlines = hlines
        self.floatfmt = floatfmt

        if self.hlines is None:
            if has_header:
                self.hlines = [0]
            else:
                self.hlines = []

        if self.n_cols is None:
            self.n_cols = len(contents[-1])

        if self.alignment is None:
            # self.alignment = 'l' + (self.n_cols - 1) * 'c'
            self.alignment = 'c' * self.n_cols
        elif len(self.alignment) == 1 and self.n_cols > 1:
            self.alignment *= self.n_cols

    def n_rows(self):
        """Return number of rows"""
        return len(self.contents)

    def table(self, caption=None, notes=None, position='h!', fontsize='small', 
              caption_above=True, **kwargs):
        """Format contents as latex table"""

        table_text = r"""
\begin{table}"""

        table_text += '[{}]'.format(position)
        table_text += r"""
\begin{center}
"""
        table_text += '\\' + fontsize + '\n'

        if caption is not None and caption_above:
            table_text += r'\caption{' + caption + '}\n'

        # write_tabular(fid, table, headers=headers, alignment=alignment, booktabs=booktabs, 
                      # floatfmt=floatfmt)
        table_text += self.tabular(**kwargs)

        table_text += r'\end{center}' + '\n'

        if caption is not None and not caption_above:
            table_text += r'\caption{' + caption + '}\n'

        if notes is not None:
            table_text += r'\footnotesize{' + notes + '}\n'

        table_text += r'\end{table}' + '\n\n'
        
        return table_text

    def tabular(self, booktabs=True):
        """Format contents as latex tabular"""

        table_text = ''
        table_text += r'\begin{tabular}{' + self.alignment + '}\n'

        tstrut = '\\rule{0pt}{2.6ex}'
        bstrut = '\\rule[-0.9ex]{0pt}{0pt}'

        if self.super_header is not None:
            table_text += '\t&'.join(self.super_header) + '\\\\\n'

        # Top rule
        if booktabs:
            table_text += r'\toprule' + '\n'
        else:
            table_text += r'\hline \hline' + '\n'

        include_tstrut = True

        # Body of table
        for i_row, row in enumerate(self.contents):

            for ii, entry in enumerate(row):
                if ii > 0:
                    table_text += '\t& '

                if isinstance(entry, float):
                    table_text += '{0:{1}}'.format(entry, self.floatfmt)
                else:
                    table_text += '{}'.format(entry)
            
            # Bottom strut
            if include_tstrut:
                table_text += tstrut
                include_tstrut = False

            if i_row in self.clines or i_row in self.hlines or i_row + 1 == self.n_rows():
                table_text += bstrut

            table_text += '\t' + r'\\' 
            table_text += ' \n'

            if i_row in self.clines:
                for (start, end) in self.clines[i_row]:
                    table_text += '\\cline{{{0}-{1}}}'.format(start, end)
                table_text += '\n'
                include_tstrut = True
                # table_text += tstrut + '%\n'
            elif i_row in self.hlines:
                if booktabs:
                    table_text += r'\midrule' + '\n'
                else:
                    table_text += r'\hline' + '\n'
                # table_text += tstrut  + '%\n'
                include_tstrut = True

        if booktabs:
            table_text += r'\bottomrule' + '\n'
        else:
            table_text += r'\hline \hline' + '\n'

        table_text += r'\end{tabular}' + '\n'

        return table_text

    def row(self, i_row):
        """Accessor for single row"""
        
        if i_row < self.n_rows():
            return self.contents[i_row]
        else:
            return self.n_cols * []

    def add_cline(self, start, end, row=0):
        """Add a partial horizontal line"""
        
        if row in self.clines:
            self.clines[row] += [(start, end)]
        else:
            self.clines[row] = [(start, end)]

        return None

    def multicolumn_row(self, text):
        """Create a multicolumn row spanning the table"""

        return [multicolumn(self.n_cols, text)]

    def add_super_header(self, text):

        self.super_header = self.multicolumn_row(text)

    # def append_multicolumn_row(self, text, before=False, is_header=False):
        # """Add a multicolumn entry spanning the table. 
        # Placed at the end, unless before=True."""

        # new_row = self.multicolumn_row(text)
        # if before:
            # self = shift_down(self, new_row, is_header)
        # else:
            # self.contents.append(new_row)

def multicolumn(n_cols, text):
    """Generate multicolumn string"""
    return '\\multicolumn{{{0}}}{{c}}{{{1}}}'.format(n_cols, text)

def hstack(table_list):
    """Stack tables horizontally"""

    n_rows = max([table.n_rows() for table in table_list])

    contents = [ut.join_lists([table.row(ii) for table in table_list]) for ii in range(n_rows)]
    n_cols = sum([table.n_cols for table in table_list])
    alignment = ''.join([table.alignment for table in table_list])
    hlines = list(set(ut.join_lists([table.hlines for table in table_list])))

    new_table = Table(contents, n_cols=n_cols, alignment=alignment, hlines=hlines)

    offset = 0
    for table in table_list:
        for row, start_end_list in table.clines.items():
            for (start, end) in start_end_list:
                new_table.add_cline(start + offset, end + offset, row)
        offset += table.n_cols

    return new_table

def empty_table(n_rows, n_cols, alignment=None):
    """Create an empty table"""

    if alignment is None:
        alignment = n_cols * 'c'

    contents = [n_cols * [' '] for row in range(n_rows)]
    return Table(contents, n_cols=n_cols, alignment=alignment)

def shift_down(table, new_row, is_header=True):

    new_table = table
    new_table.contents = [new_row] + new_table.contents

    new_table.clines = {
        key + 1 : val for key, val in table.clines.items()
    }

    new_table.hlines = [row + 1 for row in table.hlines]

    return new_table

# Join horizontally
def join_horizontal(table_list, header_list):
    """Join two tables so that each has a separate header"""

    tables_with_headers = []
    for table, header in zip(table_list, header_list):
        
        # if skip_line:
            # new_table = shift_down(table, table.n_cols * [' '])
        # else:
            # new_table = table

        new_table = table

        add_header = header is not None and table.n_cols > 1

        if add_header:
            # new_header = multicolumn(new_table.n_cols, header)
            new_header = new_table.multicolumn_row(header)
        else:
            new_header = [' ']

        new_table = shift_down(new_table, new_header)

        if add_header:
            new_table.add_cline(1, new_table.n_cols)

        tables_with_headers.append(new_table)

    n_rows = max([table.n_rows() for table in tables_with_headers])
    
    full_table_list = []
    for i_table, table in enumerate(tables_with_headers):
        full_table_list.append(table)

        added_header = header is not None and table.n_cols > 1

        if i_table < len(tables_with_headers) - 1 and added_header:
            full_table_list.append(empty_table(n_rows, n_cols=1))

    super_table = hstack(full_table_list)

    return super_table

# Append vertically
def join_vertical(table_list, header_list):
    """Append two tables with text between"""

    assert all([table.n_cols == table_list[0].n_cols for table in table_list])
    assert all([table.alignment == table_list[0].alignment for table in table_list])

    n_cols = table_list[0].n_cols
    alignment = table_list[0].alignment

    full_contents = []
    full_clines = {}
    full_hlines = []

    offset = 0
    for (table, header) in zip(table_list, header_list):

        # Add header
        if offset > 0: full_hlines.append(offset - 1)
        full_contents.append(table.multicolumn_row(header))
        full_hlines.append(offset)
        offset += 1

        # Add content
        full_contents += table.contents

        full_clines.update({
            key + offset : [(start, end) for start, end in vals]
            for key, vals in table.clines.items()
        })

        full_hlines += [val + offset for val in table.hlines]
        offset += table.n_rows()

    return Table(full_contents, n_cols, alignment=alignment, clines=full_clines, hlines=full_hlines)

# def output_tabulated(table, headers, filename, floatfmt='4.3f', tablefmt='plain'):
    # """Write table to file path"""
    # with open(filename, 'w') as fid:
        # write_tabulated(table, headers, fid, floatfmt, tablefmt)

    # return None

# def write_tabulated(table, headers, fid=None, floatfmt='4.3f', tablefmt='plain'):
    # """Write table to file stream or screen"""
    # tabulated = tabulate(table, headers, floatfmt=floatfmt, tablefmt=tablefmt)

    # if fid is None:
        # print(tabulated)
    # else:
        # fid.write(tabulated)

    # return None

def open_latex(fid):
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

    fid.write('\n' + r'\end{document}')
    return None

# TODO: define wrappers for table so that you can change floatfmt on the fly
def regression_table(results, var_names=None, vertical=False, tstat=False,
                     cov_type='HC0_se', floatfmt='4.3f', 
                     print_vars=None,
                     stats=['rsquared_adj'], **kwargs):

    tex_stats = {
        'rsquared' : 'R^2',
        'rsquared_adj' : '\\bar{R}^2'
    }

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
            contents.append(['({0:{1}})'.format(se_like[ii], floatfmt)])
        for stat in stats:
            this_stat = getattr(results, stat)
            contents.append(['{0:{1}}'.format(this_stat, floatfmt)])
    else:
        if var_names is not None:
            header_list = print_vars + [tex_stats[stat] for stat in stats]
            contents.append(header_list)
            has_header = True

        contents.append(
            [coeffs[ii] for ii in print_ix]
            + [getattr(results, stat) for stat in stats])

        contents.append(['({0:{1}})'.format(se_like[ii], floatfmt) for ii in print_ix]
                        + [' ' for stat in stats])

    return Table(contents, has_header=has_header, floatfmt=floatfmt, **kwargs)

# def write_table(fid, table, caption=None, notes=None, position='h!',
                # headers=None, alignment=None, booktabs=True, floatfmt='4.3f'):

    # fid.write(r"""
# \begin{table}""")
    # fid.write('[{}]'.format(position))
    # fid.write(r"""
# \begin{center}
# \small
# """)

    # if caption is not None:
        # fid.write(r'\caption{' + caption + '}\n')

    # write_tabular(fid, table, headers=headers, alignment=alignment, booktabs=booktabs, 
                  # floatfmt=floatfmt)

    # fid.write(r'\end{center}' + '\n')

    # if notes is not None:
        # fid.write(r'\footnotesize{' + notes + '}\n')

    # fid.write(r'\end{table}' + '\n\n')
    
    # return None

# def write_tabular(fid, contents, headers=None, alignment=None, booktabs=True,
                  # floatfmt='4.3f'):

    # all_contents = [headers] + contents
    # # table = Table(contents, headers=headers, alignment=alignment)
    # table = Table(all_contents, header=True, alignment=alignment)
    # fid.write(table.text(booktabs, floatfmt))
    # # table.write(fid, booktabs, floatfmt)
    # return None

    # """Write table to latex""" 
    # # Set alignment
    # n_cols = len(table[0])
    # if alignment is None:
        # alignment = 'l' + (n_cols - 1) * 'c'
    # elif len(alignment) == 1:
        # alignment = n_cols * alignment
        
    # fid.write(r'\begin{tabular}{' + alignment + '}\n')

    # # Top rule
    # if booktabs:
        # fid.write(r'\toprule' + '\n')
    # else:
        # fid.write(r'\hline \hline' + '\n')

    # # Headers
    # if headers is not None:

        # fid.write(' & '.join(headers) + r'\\' + '\n')
        # if booktabs:
            # fid.write(r'\midrule' + '\n')
        # else:
            # fid.write(r'\hline' + '\n')

    # # Body of table
    # for row in table:

        # assert len(row) == n_cols

        # # fid.write('$' + row[0] + '$')
        # fid.write(row[0])
        # for entry in row[1:]:
            # if isinstance(entry, float):
                # fid.write('\t& {:{floatfmt}}'.format(entry))
            # else:
                # fid.write('\t& {}'.format(entry))
        # fid.write('\t ' + r'\\' + ' \n')

    # if booktabs:
        # fid.write(r'\bottomrule' + '\n')
    # else:
        # fid.write(r'\hline \hline' + '\n')

    # fid.write(r'\end{tabular}' + '\n')

    # return None
