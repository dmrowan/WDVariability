#!/usr/bin/env python
from __future__ import print_function, division, absolute_import
import argparse
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.stats import LombScargle
from astropy.stats import median_absolute_deviation
from astropy.time import Time
import collections
import copy
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patheffects import withStroke
from matplotlib.ticker import MultipleLocator
import multiprocessing as mp
import numpy as np
import os
import pandas as pd
import _pickle as pickle
from progressbar import ProgressBar
import random
random.seed()
from statsmodels.stats.proportion import proportion_confint
import time
import WDranker_2
import WDutils
import WDVisit
import yt

def makevisitpickle(filename):
    assert(os.path.isfile(filename))
    usecols = ['t0', 't1', 't_mean',
           'mag_bgsub',
           'cps_bgsub', 'cps_bgsub_err', 'counts',
           'flux_bgsub', 'flux_bgsub_err',
           'detrad', 'flags', 'exptime']
    alldata = pd.read_csv(filename, usecols=usecols)
    #Data reduction and time correction
    alldata = WDutils.df_reduce(alldata)
    alldata = WDutils.tmean_correction(alldata)
    #Split into visits
    data = WDutils.dfsplit(alldata, 100)
    source_mag = round(np.nanmedian(alldata['mag_bgsub']),5)
    #Pull good visits
    good_indicies = []
    pbar = ProgressBar()
    for i in pbar(range(len(data))):
        df = data[i]
        visit = WDVisit.Visit(df, filename, source_mag)
        if visit.good_df() == True:
            if visit.existingperiods() == False:
                if visit.new_high_rank() == False:
                    good_indicies.append(i)

    with open(f"visitpickles/{filename}_visits.pickle", 'wb') as p:
        pickle.dump(good_indicies, p)

    print(filename, len(good_indicies))

def wrapper(path_array):
    pool = mp.Pool(processes=mp.cpu_count()+2)
    jobs = []
    for filename in path_array:
        if not os.path.isfile(f"visitpickles/{filename}_visits.pickle"):
            job = pool.apply_async(makevisitpickle, args=(filename,))
            jobs.append(job)
        else:
            print(f"Output pickle for {filename} already exists, skipping")

    for job in jobs:
        job.get()

def combineMagList(mag_array, path_array):
    visit_array = np.zeros(len(mag_array), dtype=object)
    pbar = ProgressBar()
    for i in pbar(range(len(path_array))):
        filename = path_array[i]
        assert(os.path.isfile(f"visitpickles/{filename}_visits.pickle"))
        with open(f"visitpickles/{filename}_visits.pickle", 'rb') as p:
            visit_list = pickle.load(p)
        visit_array[i] = visit_list
    MagList4 = (mag_array, path_array, visit_array)
    with open('MagList4.pickle', 'wb') as handle:
        pickle.dump(MagList4, handle)

if __name__ == '__main__':
    #makevisitpickle('ATLASJ033049.24-274737.31-NUV.csv')
    with open("MagList3.pickle", 'rb') as p:
        MagList = pickle.load(p)
    mag_array = MagList[0]
    path_array = MagList[1]

    #wrapper(path_array)
    combineMagList(mag_array, path_array)

