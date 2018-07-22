#!/usr/bin/env python
from __future__ import print_function, division
import os
import matplotlib.pyplot as plt
import argparse
import pandas as pd
import numpy as np
from WDranker_2 import catalog_match
from radec_conversion import raconvert, decconvert
#Dom Rowan REU 2018

desc="""
WDmaghist: Generate histogram of interseting sources
"""

def readBognar():
    if os.path.isfile("bognarxmatch.csv"):
        df = pd.read_csv("bognarxmatch.csv")
        g_mag = list(df['phot_g_mean_mag'])
        return g_mag
    else:
        print("generating bognar csv for xmatch")
        with open('bognar.txt') as f:
            t=f.readlines()

        ralist = []
        declist = []
        for i in range(len(t)):
            ll = len(t[i].split())
            if (ll > 15) and (ll < 25):
                l_idx = 2
                while True:
                    ra_h = t[i].split()[l_idx]
                    try:
                        ra_h = int(ra_h)
                    except:
                        l_idx += 1
                        if l_idx >= 10:
                            break
                        else:
                            continue
                    if (ra_h >= 0) and (ra_h < 24):
                        splitline = t[i].split()
                        ra_m = int(splitline[l_idx + 1])
                        ra_s = int(splitline[l_idx + 2])
                        ralist.append( (ra_h, ra_m, ra_s) )
                        
                        dec_d = splitline[l_idx + 3]
                        if dec_d == '-00':
                            negative_dec_0d = True
                        else:
                            negative_dec_0d = False
                        dec_d = int(dec_d)
                        
                        dec_m = splitline[l_idx + 4]
                        if negative_dec_0d:
                            if dec_m == '00':
                                negative_dec_0d_0m = True
                            else:
                                dec_m = int(dec_m) * -1
                                negative_dec_0d_0m = False
                        else:
                            dec_m = int(dec_m)

                        dec_s = splitline[l_idx + 5]
                        if negative_dec_0d_0m:
                            dec_s = int(dec_s) * -1
                        else:
                            dec_s = int(dec_s)

                        declist.append( (dec_d, dec_m, dec_s) )
                        break
                    else:
                        l_idx += 1

                    if l_idx >= 10:
                        break

        ralist.append( (13, 23, 50) )
        declist.append( (1,3,4) )
        ralist.append( (14,3,57) )
        declist.append( (-15,1,11) )
        ralist.append( (14, 24, 39) )
        declist.append( (9,17,14) )
        ralist.append( (19, 20, 25) )
        declist.append( (50, 17, 22) )

        ralist_deg = []
        declist_deg = []
        for tup in ralist:
            ralist_deg.append(raconvert(tup[0], tup[1], tup[2]))
        for tup in declist:
            declist_deg.append(decconvert(tup[0], tup[1], tup[2]))
        
        df_output = pd.DataFrame({'ra': ralist_deg, 'dec': declist_deg})
        df_output.to_csv("bognar_coords.csv", index=False)
        output_mag = []
        return(output_mag)

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

    mag_bognar = readBognar()


    fig, ax = plt.subplots(1,1, figsize=(6, 8))
    fig.tight_layout(rect=[.07, .03, .99, .99])
    bins = np.linspace(13, 21, 20)
    ax.hist(mag_np, color='xkcd:red', alpha=.5, 
             label="New Pulsator", zorder=2, bins=bins)
    #ax.hist(mag_kp, color='xkcd:violet', alpha=.5, 
    #         label="Known Pulastor", zorder=2, bins=bins)
    #ax.hist(mag_e, color='xkcd:azure', alpha=.5, 
    #         label="Eclipse", zorder=3, bins=bins)
    ax.hist(mag_bognar, color='xkcd:violet', alpha=.5, 
            label='Known DA Pulsators', zorder=1, bins=bins)
    ax.legend(loc=2, fontsize=15)
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

    #print(np.mean(mag_np))
    #print(np.median(mag_np))

    fig.savefig("gmaghist.pdf")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=desc)
    args= parser.parse_args()
    
    main()
