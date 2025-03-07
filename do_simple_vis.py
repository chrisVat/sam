import os
import sys
import numpy as np
import matplotlib.pyplot as plt


def main():
    project_file = "default_train_finalbest.txt"
    text_file = open(project_file, "r")
    lines = text_file.readlines()
    text_file.close()

    # remove all lines before the one that starts with: {'loss':
    for i, line in enumerate(lines):
        if line.startswith("{'loss':"):
            break
    lines = lines[i:]
    print(lines)

if __name__ == '__main__':
    main()