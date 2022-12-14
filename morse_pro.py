#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
@File    :  morse.py
@Time    :  :2022/12/10
@Author  :   Dr. Cat Lu / BFcat
@Version :   1.0
@Contact :   bfcat@live.cn
@Site    :   https://bg4xsd.github.io
@License :   (C)MIT License
@Desc    :   This is a part of project CWLab, more details can be found on the site.
'''

import os, sys

os.chdir(sys.path[0])
print("Current work directory -> %s" % os.getcwd())

from scipy import signal
import numpy as np
import random

SAMPLE_FREQ = 2000  # 2 Khz

# 60 Chars dict
MORSE_CODE_DICT = {
    'A': '.-',
    'B': '-...',
    'C': '-.-.',
    'D': '-..',
    'E': '.',
    'F': '..-.',
    'G': '--.',
    'H': '....',
    'I': '..',
    'J': '.---',
    'K': '-.-',
    'L': '.-..',
    'M': '--',
    'N': '-.',
    'O': '---',
    'P': '.--.',
    'Q': '--.-',
    'R': '.-.',
    'S': '...',
    'T': '-',
    'U': '..-',
    'V': '...-',
    'W': '.--',
    'X': '-..-',
    'Y': '-.--',
    'Z': '--..',
    '1': '.----',
    '2': '..---',
    '3': '...--',
    '4': '....-',
    '5': '.....',
    '6': '-....',
    '7': '--...',
    '8': '---..',
    '9': '----.',
    '0': '-----',
    '.': '.-.-.-',
    ',': '--..--',
    '?': '..--..',
    ':': "---...",
    ';': "-.-.-.",
    '=': '-...-',
    '+': '.-.-.',
    '-': '-....-',
    '/': '-..-.',
    '@': ".--.-.",
    '&': ".-...",
    '$': "...-..-",
    '_': "..--.-",
    '!': "-.-.--",
    '(': "-.--.",
    ')': "-.--.-",
}
ALPHABET = " " + "".join(MORSE_CODE_DICT.keys())
