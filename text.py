from tabulate import tabulate

import py_tools.utilities as ut

class Table:
    """Latex table class"""

    def __init__(self, contents, n_cols=None, header=False, alignment=None, clines=None, hlines=None):

        self.contents = contents 
        self.n_cols = n_cols
        self.header = header
        self.alignment = alignment
        self.clines = clines if clines is not None else {}
        self.hlines = hlines
        # self.vspace = vspace if vspace is not None else {}

        if self.hlines is None:
            if header:
                self.hlines = [0]
            else:
                self.hlines = []

        if self.n_cols is None:
            self.n_cols = len(contents[-1])

        # self.n_body = self.n_rows - self.n_headers

        if self.alignment is None:
            # self.alignment = 'l' + (self.n_cols - 1) * 'c'
            self.alignment = 'c' * self.n_cols
        elif len(self.alignment) == 1 and self.n_cols > 1:
            self.alignment *= self.n_cols

    def n_rows(self):
        return len(self.contents)

    def table(self, caption=None, notes=None, position='h!', fontsize='small', **kwargs):

        table_text = r"""
\begin{table}"""

        table_text += '[{}]'.format(position)
        table_text += r"""
\begin{center}
"""
        table_text += '\\' + fontsize + '\n'

        if caption is not None:
            table_text += r'\caption{' + caption + '}\n'

        # write_tabular(fid, table, headers=headers, alignment=alignment, booktabs=booktabs, 
                      # floatfmt=floatfmt)
        table_text += self.tabular(**kwargs)

        table_text += r'\end{center}' + '\n'

        if notes is not None:
            table_text += r'\footnotesize{' + notes + '}\n'

        table_text += r'\end{table}' + '\n\n'
        
        return table_text

    def tabular(self, booktabs=True, floatfmt='4.3f'):
        """Write table to latex"""

        table_text = ''

        table_text += r'\begin{tabular}{' + self.alignment + '}\n'

        tstrut = '\\rule{0pt}{2.6ex}'
        bstrut = '\\rule[-0.9ex]{0pt}{0pt}'

        # Top rule
        if booktabs:
            table_text += r'\toprule' + '\n'
        else:
            table_text += r'\hline \hline' + '\n'

        table_text += tstrut + '\n'

        # Headers
        # if self.headers is not None:

            # table_text += ' & '.join(self.headers) + r'\\' + '\n'
            # if booktabs:
                # table_text += r'\midrule' + '\n'
            # else:
                # table_text += r'\hline' + '\n'

        # Body of table
        for i_row, row in enumerate(self.contents):

            # assert i_row < self.n_headers or len(row) == self.n_cols

            # table_text += '$' + row[0] + '$'
            # table_text += row[0]
            for ii, entry in enumerate(row):
                if ii > 0:
                    table_text += '\t& '

                if isinstance(entry, float):
                    table_text += '{:{floatfmt}}'.format(entry)
                else:
                    table_text += '{}'.format(entry)
            
            # Bottom strut
            # if i_row + 1 == self.n_rows() or i_row < self.n_headers:
                # table_text += bstrut
            if i_row in self.clines or i_row in self.hlines or i_row + 1 == self.n_rows():
                table_text += bstrut

            table_text += '\t' + r'\\' 

            # if i_row in self.vspace:
                # table_text += '[{}]'.format(self.vspace[i_row])
            
            table_text += ' \n'

            if i_row in self.clines:
                for (start, end) in self.clines[i_row]:
                    table_text += '\\cline{{{0}-{1}}}'.format(start, end)
                table_text += '\n'
                table_text += tstrut + '%\n'
            elif i_row in self.hlines:
                if booktabs:
                    table_text += r'\midrule' + '\n'
                else:
                    table_text += r'\hline' + '\n'
                table_text += tstrut  + '%\n'

        if booktabs:
            table_text += r'\bottomrule' + '\n'
        else:
            table_text += r'\hline \hline' + '\n'

        table_text += r'\end{tabular}' + '\n'

        return table_text

    def row(self, i_row):
        
        if i_row < self.n_rows():
            return self.contents[i_row]
        else:
            return self.n_cols * []

    def add_cline(self, start, end, row=0):
        
        if row in self.clines:
            self.clines[row] += [(start, end)]
        else:
            self.clines[row] = [(start, end)]

        return None

    # def add_vspace(self, space, row=0):
        # self.vspace[row] = space

def hstack(table_list):
    """Stack tables horizontally"""

    # assert all([table.n_headers == table_list[0].n_headers for table in table_list])

    n_rows = max([table.n_rows() for table in table_list])

    contents = [ut.join_lists([table.row(ii) for table in table_list]) for ii in range(n_rows)]
    n_cols = sum([table.n_cols for table in table_list])
    # n_headers = table_list[0].n_headers # TODO: generalize
    alignment = ''.join([table.alignment for table in table_list])
    hlines = list(set(ut.join_lists([table.hlines for table in table_list])))

    new_table = Table(contents, n_cols=n_cols, alignment=alignment, hlines=hlines)

    offset = 0
    for table in table_list:
        for row, start_end_list in table.clines.items():
            for (start, end) in start_end_list:
                new_table.add_cline(start + offset, end + offset, row)
        offset += table.n_cols

    # n_rows = max(left.n_rows(), right.n_rows())
    # assert left.n_headers == right.n_headers

    # contents = [left.row(ii) + right.row(ii) for ii in range(n_rows)]
    # n_cols = left.n_cols + right.n_cols
    # n_headers = left.n_headers
    # alignment = left.alignment + right.alignment

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
    # if is_header:
        # new_table.n_headers += 1

    new_table.clines = {
        key + 1 : val for key, val in table.clines.items()
    }

    new_table.hlines = [row + 1 for row in table.hlines]

    return new_table

# Join horizontally
def join_subtables(table_list, header_list):
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
            new_header = '\\multicolumn{{{0}}}{{c}}{{{1}}}'.format(new_table.n_cols, header)
        else:
            new_header = ' '

        # if skip_line:
            # \vspace{}

        new_table = shift_down(new_table, [new_header])

        if add_header:
            new_table.add_cline(1, new_table.n_cols)

        tables_with_headers.append(new_table)
        # new_table = table
        # new_table.contents = ['\\multicolumn{{{0}}}{{c}}{{{1}}}'.format(new_table.n_cols, header), new_table.contents]
        # new_table.n_headers += 1
        # tables_with_headers.append(new_table)

    n_rows = max([table.n_rows() for table in tables_with_headers])
    
    full_table_list = []
    for i_table, table in enumerate(tables_with_headers):
        full_table_list.append(table)

        added_header = header is not None and table.n_cols > 1

        if i_table < len(tables_with_headers) - 1 and added_header:
            full_table_list.append(empty_table(n_rows, n_cols=1))
        # full_table_list += [empty_table(n_rows, n_cols=1, n_headers=tables_with_headers[0].n_headers), table]

    super_table = hstack(full_table_list)
    # if space is not None:
        # super_table.add_vspace(space)

    return super_table

# Append vertically
# def append_subtables(table_list, header_list):
    # """Join two tables with text between"""

def output_tabulated(table, headers, filename, floatfmt='4.3f', tablefmt='plain'):
    """Write table to file path"""
    with open(filename, 'w') as fid:
        write_tabulated(table, headers, fid, floatfmt, tablefmt)

    return None

def write_tabulated(table, headers, fid=None, floatfmt='4.3f', tablefmt='plain'):
    """Write table to file stream or screen"""
    tabulated = tabulate(table, headers, floatfmt=floatfmt, tablefmt=tablefmt)

    if fid is None:
        print(tabulated)
    else:
        fid.write(tabulated)

    return None

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

def write_table(fid, table, caption=None, notes=None, position='h!',
                headers=None, alignment=None, booktabs=True, floatfmt='4.3f'):

    fid.write(r"""
\begin{table}""")
    fid.write('[{}]'.format(position))
    fid.write(r"""
\begin{center}
\small
""")

    if caption is not None:
        fid.write(r'\caption{' + caption + '}\n')

    write_tabular(fid, table, headers=headers, alignment=alignment, booktabs=booktabs, 
                  floatfmt=floatfmt)

    fid.write(r'\end{center}' + '\n')

    if notes is not None:
        fid.write(r'\footnotesize{' + notes + '}\n')

    fid.write(r'\end{table}' + '\n\n')
    
    return None

def write_tabular(fid, contents, headers=None, alignment=None, booktabs=True,
                  floatfmt='4.3f'):

    all_contents = [headers] + contents
    # table = Table(contents, headers=headers, alignment=alignment)
    table = Table(all_contents, header=True, alignment=alignment)
    fid.write(table.text(booktabs, floatfmt))
    # table.write(fid, booktabs, floatfmt)
    return None

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
