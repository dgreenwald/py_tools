#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 14:34:39 2025

@author: dan
"""

import os
from py_tools import parsing as par
import re
from collections import namedtuple

Command = namedtuple('Command', ['name', 'nargs', 'content'])

def get_next_argument(text):
    
    match = re.match(r'\s*{', text)
    bma_init = par.match_to_bma(match, text)
    if not bma_init.matched:
        raise Exception
    
    argument, _, after = par.expand_to_target(bma_init.after, left_bracket='{')
    
    return argument, after

def get_arguments(text, nargs):
    
    arguments = []
    for ii in range(nargs):
        argument, text = get_next_argument(text)
        arguments.append(argument)
        
    return arguments, text

def replace_content(text, command):
    
    # If we go to double digits there will be issues with the regex, where #10 catches #1, etc.
    assert command.nargs < 10
    
    new_content = command.content
    arguments, text = get_arguments(text, command.nargs)
    assert len(arguments) == command.nargs
    for ii, argument in enumerate(arguments):
        arg_no = ii + 1
        new_content = re.sub(rf'#{arg_no}', re.escape(argument), new_content)
        
    return new_content, text

def replace_command(text, command):
    
    pattern = r'\\' + command.name + r'(?=[\W_])'
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
    
    # Loop through until we stop finding any commands
    done = False
    while not done:
        done = True
        
        # Loop through commands
        for command in commands:
            text, replaced_any = replace_command(text, command)
            done = done and (not replaced_any)
            
    return text

def remove_comments(text, replaced_text=''):
    
    pattern = r'(?<!\\)%'
    bma = par.bma_search(pattern, text)
    if not bma.matched:
        return text
    
    # Add 
    replaced_text += bma.before
    eol_pattern = r'\n'
    bma_eol = par.bma_search(eol_pattern, bma.after)
    if bma_eol.matched:
        # We found the end of the line, proceed from there
        replaced_text += remove_comments(bma_eol.after)
    
    return replaced_text

def get_definitions(text):
    
    defn_pattern = r'^\s*\\def\s*\\(\w*)\s*{\s*'
    
    defn_list = []
    for match in re.finditer(defn_pattern, text, re.MULTILINE):
        
        bma = par.match_to_bma(match, text)
        name = bma.match.groups(1)[0]
        content, _, _ = par.expand_to_target(bma.after, left_bracket='{')
        
        defn_list.append(Command(name, 0, content))
        
    return defn_list

def remove_command(text, command):
    
    # Name of command
    pattern = r'\\newcommand\s*{\s*\\' + re.escape(command.name) + r'}'
    
    # Number of arguments
    if command.nargs > 0:
        pattern += fr'\s*\[{command.nargs:d}\]'
    
    # Content
    pattern += r'\s*{' + re.escape(command.content) + r'\s*}\s*\n'
    
    # Replace
    text = re.sub(pattern, '', text)
    
    return text

def remove_commands(text, command_list):
    
    for command in command_list:
        text = remove_command(command)
        
    return text

def remove_defn(text, command):
    
    # Name of command
    pattern = r'\\def\s*\\' + re.escape(command.name)
    
    # Content
    pattern += r'\s*{' + re.escape(command.content) + r'\s*}\s*'
    
    # Replace
    text = re.sub(pattern, '', text)
    
    return text

def remove_defns(text, defn_list):
    
    for defn in defn_list:
        text = remove_defn(defn)
        
    return text

def replace_ref(text, label, replacement):
    
    pattern = r'\\ref\s*{\s*' + label + r'}'
    text = re.sub(pattern, replacement, text)
    
    pattern = r'\\eqref\s*{\s*' + label + r'}'
    text = re.sub(pattern, '(' + replacement + ')', text)
    return text

def replace_refs(text, refs_to_replace):
    
    for label, replacement in refs_to_replace:
        text = replace_ref(text, label, replacement)
        
    return text

def replace_figures(text, figs_to_replace):
    
    for orig, repl in figs_to_replace:
        pattern = re.escape(orig)
        text = re.sub(pattern, repl, text)
        
    return text

def get_commands(text):
    
    command_pattern = r'^\s*\\newcommand{\s*\\'
    nargs_pattern = r'\[\s*(\d*)\s*\]'
    start_bracket_pattern = r'\s*{'
    
    command_list = []
    for match in re.finditer(command_pattern, text, re.MULTILINE):
        
        bma = par.match_to_bma(match, text)
        name, _, after = par.expand_to_target(bma.after, left_bracket='{')
            
        # Get number of arguments
        match_nargs = re.match(nargs_pattern, after)
        if match_nargs is None:
            # No arguments
            nargs = 0
        else:
            # Has argument number
            # print(match_nargs.groups(1))
            nargs = match_nargs.groups(1)[0]
            if nargs == '':
                nargs = 0
            else:
                nargs = int(nargs)
            # nargs = int(match_nargs.groups(1)[0])
            
            # Skip past this
            after = after[match_nargs.end():]
            
        match_start_bracket = re.match(start_bracket_pattern, after)
        assert (match_start_bracket is not None)
        after = after[match_start_bracket.end():]
        content, _, _ = par.expand_to_target(after, left_bracket='{')
            
        command_list.append(Command(name, nargs, content))
        
    return command_list
    
def read_if_exists(filepath):
    
    if os.path.exists(filepath):
        with open(filepath, 'rt') as fid:
            return fid.read()
    else:
        return None
    
def remove_brackets(text):
    
    bma = par.bma_search(r'{', text)
    if bma.matched:
        argument, _, after = par.expand_to_target(bma.after, left_bracket='{')
        argument_new = remove_brackets(argument)
        after_new = remove_brackets(after)
        return bma.before + argument_new + after_new
    else:
        return text
    
def read_input_file(filepath):
    
    # filepath = filepath_in.strip()
    
    filepath = remove_brackets(filepath)
    
    # Try to load in file
    text = read_if_exists(filepath)
    if text is not None:
        return text
    
    # If we failed, try adding the .tex extension
    backup_path = filepath + '.tex'
    text = read_if_exists(backup_path)
    
    return text

def remove_unused_commands(text, command_list=None, defn_list=None):
    
    if command_list is None: command_list = []
    if defn_list is None: defn_list = []
    
    done = False
    while not done:
        
        done = True
        for command in command_list:
            matches = re.findall(r'\\' + command.name + r'\b', text)
            if len(matches) == 1:
                done = False
                text = remove_command(text, command)
                
        for defn in defn_list:
            matches = re.findall(r'\\' + defn.name + r'\b', text)
            if len(matches) == 1:
                done = False
                text = remove_defn(text, defn)
            
    return text

def get_aux_labels(aux_file):
    
    with open(aux_file, 'rt') as fid:
        aux_lines = fid.readlines()
        
    refs_to_replace = []
    for line in aux_lines:
        pattern = r'\\newlabel{'
        bma = par.bma_search(pattern, line)
        if bma.matched:
            tag, matched_bracket, after = par.expand_to_target(bma.after, left_bracket='{')
            label_match = re.match(r'{{(\w+\.?\w*)}', after)
            label = label_match.groups(1)[0]
            refs_to_replace.append((tag, label))
            
    return refs_to_replace

def replace_commands_dynamic(text, commands_to_replace, names_to_replace=None):
    
    current_names = [cmd.name for cmd in commands_to_replace]
    if names_to_replace is None:
        names_to_replace = []
    
    # Need to dynamically update the list of definitions, so only replace until next \def
    defn_pattern = r'^\s*\\def\s*\\(\w*)\s*{\s*'
    bma = par.bma_search(defn_pattern, text, re.MULTILINE)
    
    # Replace up to this point using current list
    before = replace_commands_static(bma.before, commands_to_replace)
    
    if bma.matched:
        # If we have a new definition, need to update the list
        commands_to_replace_new = commands_to_replace.copy()
        
        # Get the new definition name
        name = bma.match.groups(1)[0]
        
        # Expand through the argument
        content, bracket, after = par.expand_to_target(bma.after, left_bracket='{')
        
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
            
            before += content + bracket

        # Now replace the rest of the text
        after = replace_commands_dynamic(
            after, commands_to_replace_new, names_to_replace=names_to_replace
            )
            
    else:
        # Otherwise, we are done
        after = ''
        
    text_new = before + after
    
    return text_new

# def replace_body_definition(text, name, content_current=None, replaced_text=''):
    
#     defn_pattern = r'^\s*\\def\s*\\' + re.escape(name) + '\s*{\s*'
#     bma = par.bma_search(defn_pattern, text, re.MULTILINE)
    
#     before = bma.before
#     if content_current is not None:
#         before = replace_command()
        
#     content_new, _, after = par.expand_to_target(bma.after, left_bracket='{')

# def replace_body_definitions(text):
    
#     defns = get_definitions(text)
#     names = [defn.name for defn in defns]
    
    
#     # for ii, defn in enumerate(defns):
#     #     start_point = 
    
def flatten_text(text, flattened_text='', commands_to_replace=None, 
                 # defns_to_replace=None, 
                 remove_comments_from_text=True, 
                 names_to_replace=None):
    
    # Replace any definitions in the body of the text
    if commands_to_replace is None:
        commands_to_replace = []
    
    if remove_comments_from_text:
        text = remove_comments(text)
        
    text = replace_commands_dynamic(text, commands_to_replace,
                                    names_to_replace=names_to_replace)
    
    pattern = r'\\input\s*{'
    
    bma = par.bma_search(pattern, text)
    if not bma.matched:
        flattened_text += text
        return flattened_text
    
    # Add text up to input
    flattened_text += bma.before
    
    # Capture text inside brackets
    argument, matched_bracket, after_input = par.expand_to_target(bma.after, left_bracket='{')
    
    # Try to read from file
    filepath = argument.strip()
    input_text = read_input_file(filepath)
    
    if input_text is not None:
        # print(f'loaded {filepath}')
        # It worked, load in the text (after recursively flattening)
        flattened_text += flatten_text(
            input_text, commands_to_replace=commands_to_replace,
            # defns_to_replace=defns_to_replace,
            names_to_replace=names_to_replace,
            )
    else:
        # It didn't work, leave it as-is
        print(f'failed to load {filepath}')
        flattened_text += bma.middle + argument + matched_bracket

    # Continue through rest of the file
    flattened_text += flatten_text(after_input, commands_to_replace=commands_to_replace,
                                   # defns_to_replace=defns_to_replace,
                                   names_to_replace=names_to_replace)
            
    return flattened_text

def flatten(infile=None, outfile=None, names_to_replace=None, 
            aux_reference_file=None, remove_comments_from_text=True, 
            remove_comments_from_preamble=False, remove_unused=False):
    
    if names_to_replace is None:
        names_to_replace = []
    
    # if text is None:
        
    #     assert infile is not None
        
    head, tail = os.path.split(infile)
    if head != '':
        os.chdir(head)

    with open(infile, 'rt') as fid:
        text = fid.read()

    bma_preamble = par.bma_search(r'\\begin\s*{\s*document\s*}', text)
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
        bma_preamble.after, commands_to_replace=commands_and_defns_to_replace,
        # defns_to_replace=defns_to_replace,
        remove_comments_from_text=remove_comments_from_text,
        names_to_replace=names_to_replace,
        )

    # Remove preamble comments?
    preamble_new = bma_preamble.before
    if remove_comments_from_preamble:
        preamble_new = remove_comments(preamble_new)

    # Reassemble
    text = preamble_new + bma_preamble.middle + after_preamble_new
    
    # Replace references from auxiliary document
    if aux_reference_file is not None:
        
        refs_to_replace = get_aux_labels(aux_reference_file)
        text = replace_refs(text, refs_to_replace)
        
    # Get rid of unneeded commands
    if remove_unused:
        text = remove_unused_commands(text, command_list=command_list, defn_list=defn_list)
    
    if outfile is not None:
        with open(outfile, 'wt') as fid:
            fid.write(text)
        return None
            
    return text