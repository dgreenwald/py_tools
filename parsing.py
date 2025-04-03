#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 29 21:59:19 2025

@author: dan
"""

from collections import namedtuple
import re

BmaMatch = namedtuple('BmaMatch', ['before', 'middle', 'after', 'match', 'matched'])

def match_to_bma(match, text):
    
    if match is None:
        before = text
        middle = ''
        after = ''
        matched = False
    else:
        before = text[:match.start()]
        middle = text[match.start():match.end()]
        after = text[match.end():]
        matched = True
        
    return BmaMatch(before, middle, after, match, matched)

def bma_search(pattern, text):
    
    match = re.search(pattern, text)
    return match_to_bma(match, text)
    # if match is None:
    #     before = text
    #     middle = ''
    #     after = ''
    #     matched = False
    # else:
    #     before = text[:match.start()]
    #     middle = text[match.start():match.end()]
    #     after = text[match.end():]
    #     matched = True
        
    # return BmaMatch(before, middle, after, match, matched)

def expand_to_target(text, target=None, depth=1, target_depth=1, left_bracket='(', right_bracket=')',
                     before_text=''):
    
    if left_bracket == '(':
        right_bracket = ')'
    elif left_bracket == '[':
        right_bracket = ']'
    elif left_bracket == '{':
        right_bracket = '}'
    else:
        raise Exception
        
    if target is None:
        target = right_bracket
        
    pattern = rf'([{left_bracket}{right_bracket}]|{re.escape(target)})'
        
    bma = bma_search(pattern, text)
    if not bma.matched:
        raise Exception
        
    # match = re.search(pattern, text)
    # if match is None:
    #     raise Exception
        
    # before_text += text[:match.start()]
    # matched_text = text[match.start():match.end()]
    # after_text = text[match.end():]
    
    before_text += bma.before
        
    if (target in bma.match.group(0)):
        # We found the target
        if (target_depth is None) or (depth == target_depth):
            # We are at the right depth level, return
            return before_text, bma.middle, bma.after
        else:
            # We are at the wrong depth
            new_depth = depth
            
    # We found a bracket
    if (left_bracket in bma.match.group(0)):
        # We found a subparen, increase the depth by 1
        new_depth = depth + 1
    elif (right_bracket in bma.match.group(0)):
        # We closed a paren, decrease the depth by 1
        new_depth = depth - 1
        
    # If we failed to finish, fold the match into the before text
    before_text += bma.middle
        
    # Recurse further
    return expand_to_target(
        bma.after, target=target, depth=new_depth, 
        left_bracket=left_bracket, right_bracket=right_bracket,
        before_text=before_text,
        )