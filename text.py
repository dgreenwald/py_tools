from tabulate import tabulate

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

def write_table(fid, table, caption=None, notes=None, 
                headers=None, alignment=None, booktabs=True, floatfmt='4.3f'):

    fid.write(r"""
\begin{table}[h!]
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

def write_tabular(fid, table, headers=None, alignment=None, booktabs=True,
                  floatfmt='4.3f'):

    """Write table to latex""" 
    # Set alignment
    n_cols = len(table[0])
    if alignment is None:
        alignment = 'l' + (n_cols - 1) * 'c'
    elif len(alignment) == 1:
        alignment = n_cols * alignment
        
    fid.write(r'\begin{tabular}{' + alignment + '}\n')

    # Top rule
    if booktabs:
        fid.write(r'\toprule' + '\n')
    else:
        fid.write(r'\hline' + '\n')

    # Headers
    if headers is not None:

        fid.write(' & '.join(headers) + r'\\' + '\n')
        if booktabs:
            fid.write(r'\midrule' + '\n')
        else:
            fid.write(r'\hline' + '\n')

    # Body of table
    for row in table:

        assert len(row) == n_cols

        # fid.write('$' + row[0] + '$')
        fid.write(row[0])
        for entry in row[1:]:
            if isinstance(entry, float):
                fid.write('\t& {:{floatfmt}}'.format(entry))
            else:
                fid.write('\t& {}'.format(entry))
        fid.write('\t ' + r'\\' + ' \n')

    if booktabs:
        fid.write(r'\bottomrule' + '\n')
    else:
        fid.write(r'\hline' + '\n')

    fid.write(r'\end{tabular}' + '\n')

    return None
