#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
@File    :  MorseSoundGen.py
@Time    :  :2022/12/04
@Author  :   Dr. Cat Lu / BFcat
@Version :   1.0
@Contact :   bfcat@live.cn
@Site    :   https://bg4xsd.github.io
@License :   (C)MIT License
@Desc    :   This is a part of project CW Terminator, more details can be found on the site.

'''

# This code is adopted and modified from below URLS:
# 1. https://compucademy.net/morse-code-with-python/
# 2. https://codereview.stackexchange.com/questions/259634/simple-morse-code-converter-python
# 3. https://code.activestate.com/recipes/578411-a-complete-morse-code-generator-in-python-with-sou/
# 4. https://www.geeksforgeeks.org/morse-code-translator-python/

import os
import sys
import pygame
import time

os.chdir(sys.path[0])
print("Current work directory -> %s" % os.getcwd())

ENGLISH_TO_MORSE = {
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
    '0': '-----',
    '1': '.----',
    '2': '..---',
    '3': '...--',
    '4': '....-',
    '5': '.....',
    '6': '-....',
    '7': '--...',
    '8': '---..',
    '9': '----.',
}

TIME_BETWEEN = 0.5  # Time between sounds
PATH = "../morse_code_audio/"


def verify(string):
    keys = list(ENGLISH_TO_MORSE.keys())
    for char in string:
        if char not in keys and char != " ":
            print(f"The character {char} cannot be translated.")
            raise SystemExit


def main():
    print("### English to Morse Code Audio Converter ###")
    print("Enter your message in English: ")
    message = input("> ").upper()
    verify(message)

    pygame.init()

    for char in message:
        if char == " ":
            print(" " * 3, end=" ")  # Separate words clearly
            time.sleep(7 * TIME_BETWEEN)
        else:
            print(ENGLISH_TO_MORSE[char.upper()], end=" ")
            pygame.mixer.music.load(
                PATH + char + '_morse_code.ogg'
            )  # You will need these sound filest
            pygame.mixer.music.set_volume(0.1)
            pygame.mixer.music.play()
            time.sleep(3 * TIME_BETWEEN)


if __name__ == "__main__":
    main()
