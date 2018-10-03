#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
A set of helper functions for preprocessing or visualization.
'''

import json
import arrow
import folium
import webbrowser
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

def load_geojson(geojson_path):
    '''
    load geojson file into a python dict, key is the beat name, and value is the
    polygon with coordinates.
    '''
    geojson_obj = {}
    with open(geojson_path, 'r') as f:
        data = json.load(f)
    for feature in data['features']:
        beat        = feature['properties']['BEAT']
        type        = feature['geometry']['type']
        coordinates = feature['geometry']['coordinates']
        if type == 'Polygon':
            geojson_obj[beat] = Polygon(coordinates[0])
        elif type == 'MultiPolygon':
            geojson_obj[beat] = Polygon(coordinates[0][0])
    return geojson_obj

def proj2beats(s, geojson_path):
    '''
    project spatio points into discrete beats area by checking which beats the
    points are located in.
    '''
    # max_x, min_x = s[:, 0].max(), s[:, 0].min()
    # max_y, min_y = s[:, 1].max(), s[:, 1].min()
    # print('max x %f, min x %f.' % (max_x, min_x))
    # print('max y %f, min y %f.' % (max_y, min_y))
    # load geojson
    geojson_obj = load_geojson(geojson_path)
    # assign the beat where points occurred.
    beats = [ 'Invalid_beat' for i in range(len(s)) ]
    for i in range(len(s)):
        for beat in geojson_obj:
            # if Point(*point).within(geojson_obj[beat]):
            if geojson_obj[beat].contains(Point(*s[i])):
                beats[i] = beat
                break
    # tokenize the beats representation
    beats_set = list(set(beats) - {'Invalid_beat'})
    beats_set.sort()
    beats_set.append('Invalid_beat')
    beats = [ beats_set.index(beat) for beat in beats ]
    return beats_set, np.array(beats)

def plot_intensities4beats(
        Mu, beats_set,
        geojson_path='/Users/woodie/Desktop/workspace/Zoning-Analysis/data/apd_beat.geojson',
        html_path='intensity_map.html',
        center=[33.796480,-84.394220]):
    '''plot background rate intensities over a map.'''
    intensities = DataFrame(list(zip(beats_set, Mu)), columns=['BEAT', 'intensity'])
    map = folium.Map(location=center, zoom_start=13, zoom_control=True, max_zoom=17, min_zoom=10)
    map.choropleth(
        geo_data=open(geojson_path).read(),
        data=intensities,
        columns=['BEAT', 'intensity'],
        key_on='feature.properties.BEAT',
        fill_color='YlGn',
        fill_opacity=0.7,
        line_opacity=0.2,
        highlight=True,
        legend_name='Background intensities'
    )
    folium.LayerControl().add_to(m)
    map.save(html_path)

def load_police_training_data(n=500):
    '''load police training data from local files.'''
    # data path
    geojson_path = '/Users/woodie/Desktop/workspace/Zoning-Analysis/data/apd_beat.geojson'
    points_path  = 'data/10k.points.txt'
    marks_path   = 'resource/embeddings/10k.gbrbm.hid1k.txt'
    labels_path  = 'data/10k.labels.txt'
    # load data
    print('[%s] loading training data...' % arrow.now())
    labels = []
    points = np.loadtxt(points_path, delimiter=',')
    marks  = np.loadtxt(marks_path, delimiter=',')
    with open(labels_path, 'r') as f:
        labels = [ line.strip() for line in f ]
    # prepare training data
    print('[%s] preparing training data...' % arrow.now())
    t = points[:, 0]      # time sequence
    s = points[:, [2, 1]] # spatio sequence
    # reorder the sequence in training data
    t_order = t.argsort()
    t = t[t_order]
    s = s[t_order]
    m = marks[t_order] # marks sequence
    l = [ labels[i] for i in t_order ]
    # test partially
    t = t[:n]
    s = s[:n, ]
    m = m[:n, ]
    l = l[:n]
    print('[%s] %d data points collected, as well as %d existed types of labels.' % \
        (arrow.now(), len(t), len(set(l))))
    print('[%s] time of the data ranges from %d to %d.' % (arrow.now(), t[0], t[-1]))
    # project spatio sequence into beats area
    print('[%s] projecting spatial points into beat area...' % arrow.now())
    u_set, u = proj2beats(s, geojson_path)
    print('[%s] %d beats were found in the dataset, %d of them are invalid beats.' % \
        (arrow.now(), len(u_set), len(u[u==len(u_set)-1])))
    return t, m, l, u, u_set
