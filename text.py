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

def write_tabular(fid, table, headers=None, alignment=None, booktabs=True,
                  floatfmt='4.3f'):

    """Write table to latex""" 

    fid.write(r'\begin{tabular}')

    # Set alignment
    n_cols = len(table[0])
    if alignment is None:
        alignment = 'l' + (n_cols - 1) * 'c'
    elif len(alignment) == 1:
        alignment = n_cols * alignment
        
    fid.write('{' + alignment + '}\n')

    # Top rule
    if booktabs:
        fid.write(r'\toprule')
    else:
        fid.write(r'\hline')
    fid.write('\n')

    # Headers
    if headers is not None:

        fid.write(' & '.join(headers) + r'\\')
        fid.write('\n')
        if booktabs:
            fid.write(r'\midrule')
        else:
            fid.write(r'\hline')

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
        fid.write(r'\bottomrule')
    else:
        fid.write(r'\hline')

    fid.write(r'\end{tabular}')

    return None
