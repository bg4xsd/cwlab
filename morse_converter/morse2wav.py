#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
@File    :  morse2wav.py
@Time    :  :2022/12/04
@Author  :   Dr. Cat Lu / BFcat
@Version :   1.0
@Contact :   bfcat@live.cn
@Site    :   https://bg4xsd.github.io
@License :   (C)MIT License
@Desc    :   This is a part of project CW Converter, more details can be found on the site.
'''


import os
import sys
import pygame
import time

import morse as morse
import morse_sound as ms

os.chdir(sys.path[0])
print("Current work directory -> %s" % os.getcwd())
