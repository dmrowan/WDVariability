#!/usr/bin/env python
from __future__ import print_function, division
import os
import matplotlib.pyplot as plt
import argparse
import pandas as pd
import numpy as np
from WDranker_2 import catalog_match
#Dom Rowan REU 2018

desc="""
WDmaghist: Generate histogram of interseting sources
"""

def main():
    assert(os.getcwd() == '/home/dmrowan/WhiteDwarfs/InterestingSources')
    assert(os.path.isfile("IS.csv"))

    df = pd.read_csv('IS.csv')
    idx_drop = []
    for i in range(len(df['g'])):
        if str(df['g'][i]) == 'nan':
            idx_drop.append(i)
    df = df.drop(index=idx_drop)
    df = df.reset_index(drop=True)

    mag_kp = []
    mag_np = []
    mag_e = []
    for i in range(len(df['g'])):
        if df['type'][i] == 'KnownPulsator':
            mag_kp.append(df['g'][i])
        elif df['type'][i] == 'Pulsator':
            mag_np.append(df['g'][i])
        else:
            mag_e.append(df['g'][i])


    fig, ax = plt.subplots(1,1, figsize=(8, 12))
    ax.hist(mag_np, color='xkcd:red', alpha=.5, 
             label="New Pulsator", zorder=1)
    ax.hist(mag_kp, color='xkcd:violet', alpha=.5, 
             label="Known Pulastor", zorder=2)
    ax.hist(mag_e, color='xkcd:azure', alpha=.5, 
             label="Eclipse", zorder=3)
    ax.legend(loc=2, fontsize=25)
    ax.minorticks_on()
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    ax.tick_params(direction='in', which='both', labelsize=15)
    ax.tick_params('both', length=8, width=1.8, which='major')
    ax.tick_params('both', length=4, width=1, which='minor')
    ax.set_xlabel('Gaia apparent G magnitude (mag)', fontsize=20)
    ax.set_ylabel('Number in bin', fontsize=20)

    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(1.5)

    fig.savefig("gmaghist.pdf")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=desc)
    args= parser.parse_args()
    
    main()
