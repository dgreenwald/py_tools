import re

def replace_camel(oldfile, newfile):

    p_camel = re.compile('def ([a-z0-9_]*)([A-Z])(\w*)')

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

replace_camel('in_out.py', 'in_out_test.py')
