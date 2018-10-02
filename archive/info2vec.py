#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This is the main script for generating vectors for spatio-temporal information.
"""

import sys
import arrow
import numpy as np

def clean_data(raw_info, raw_text):
    indices = []
    with open(raw_info, 'r') as fr, \
         open('data/rawdata/clean_random_cases_info.txt', 'w') as fw:
        idx = 0
        for line in fr:
            try:
                data = line.strip().split('\t')
                _id      = data[0]
                code     = data[1]
                category = data[2].strip()
                date     = data[3]
                stime    = data[4]
                etime    = data[5]
                lat      = data[14]
                lng      = data[15]
                if category != '' and category != 'NON-CRIMINAL REPORT':
                    indices.append(idx)
                    fw.write('\t'.join((stime, lat, lng, _id, code, category)) + '\n')
            except Exception as e:
                print(e)
            idx += 1
        print(len(indices))

    with open(raw_text, 'r') as fr, \
         open('data/rawdata/clean_random_cases_text.txt', 'w') as fw:
        docs = [ line for line in fr ]
        for idx in indices:
            fw.write(docs[idx])

def info2vec(filename):
    counter = 0
    points  = []
    with open(filename, 'r') as f:
        for line in f:
            # prefix 'r' stands for raw (feature)
            rtime, rlat, rlng = line.strip().split('\t')
            rtime = rtime.strip()

            if ('T' in rtime) or (' ' in rtime):
                time = arrow.get(rtime)
            else:
                time = [ int(t) for t in rtime.split('-') ]
                time = arrow.Arrow(*time)
            time = time.timestamp
            lat  = float(rlat) / 1e+5
            lng  = -1 * float(rlng) / 1e+5
            points.append([time, lat, lng])

    points = np.array(points)
    # print(points)
    return points

if __name__ == '__main__':
    labeled_path = 'data/rawdata/56_labeled_cases_info.txt'
    random_path  = 'data/rawdata/129k_random_cases_info.txt'
    # clean_data()
    labeled_points = info2vec(labeled_path)
    random_points  = info2vec(random_path)
    points = np.concatenate([labeled_points, random_points], axis=0)
    print(points.shape)
    np.savetxt("data/129k.points.txt", points, delimiter=',', fmt='%1.5f')
