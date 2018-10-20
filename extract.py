#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This is the main script for generating embeddings by RegRBM.
"""
from __future__ import print_function

import sys
import numpy as np

with open('data/129k.labels.txt', 'r') as fl, \
     open('data/129k.labels.txt', 'r') as fl1, \
     open('data/129k.points.txt', 'r') as fp, \
     open('data/129k.text.txt', 'r') as ft, \
     open('data/meta/burglary_set.txt', 'r') as fbs, \
     open('data/meta/robbery_set.txt', 'r') as frs, \
     open('data/burglary/all.burglary.labels.txt', 'w') as fbw, \
     open('data/robbery/all.robbery.labels.txt', 'w') as frw:

    burglary_set = [ line.strip('\n') for line in fbs ]
    robbery_set  = [ line.strip('\n') for line in frs ]

    burglary_set = ['burglary'] + burglary_set
    robbery_set  = ['pedrobbery', 'DIJAWAN_ADAMS', 'JAYDARIOUS_MORRISON', 'JULIAN_TUCKER', 'THADDEUS_TODD'] + robbery_set

    print(burglary_set)
    print(robbery_set)

    i_burglary_set = []
    i_robbery_set  = []
    i            = 0
    for line in fl:
        label = line.strip('\n')
        if label in burglary_set:
            i_burglary_set.append(i)
        elif label in robbery_set:
            i_robbery_set.append(i)
        i += 1

    print(len(i_burglary_set))
    print(len(i_robbery_set))

    j = 0
    for line in fl1:
        if j in i_burglary_set:
            fbw.write(line)
        elif j in i_robbery_set:
            frw.write(line)
        j += 1
