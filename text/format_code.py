import re

def replace_camel(oldfile, newfile):
    """Rename camelCase function definitions to snake_case in a Python source file.

    Reads ``oldfile``, finds all ``def`` statements whose names contain
    camelCase segments, and writes a converted version to ``newfile``.  The
    renaming is applied globally (not just in ``def`` lines) so that call
    sites are updated as well.  The process repeats until no more camelCase
    names are found.

    Parameters
    ----------
    oldfile : str
        Path to the source Python file to convert.
    newfile : str
        Path where the converted file will be written.  May be the same as
        ``oldfile`` to convert in place (the function always reads before
        writing).

    Returns
    -------
    None
    """
    p_camel = re.compile(r'def ([a-z0-9_]*)([A-Z])(\w*)')

    done = False
    readfile = oldfile
    writefile = newfile

    while not done:

        done = True
        replace_list = []

        with open(readfile, 'r') as fid:
            for line in fid:
               matches = re.findall(p_camel, line)
               if matches:
                   done = False
                   for match in matches:
                        oldname = ''.join(match)
                        newname = match[0] + '_' + match[1].lower() + match[2]
                        replace_list.append((oldname, newname))
        
        if not done:
            write_lines = []
            with open(readfile, 'r') as fid:
                for line in fid:
                    newline = line
                    for oldname, newname in replace_list:
                        newline = re.sub(oldname, newname, newline)
                    write_lines.append(newline)

            with open(writefile, 'w') as fid:
                fid.writelines(write_lines)

            readfile = newfile
