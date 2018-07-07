#!/usr/bin/env python
from __future__ import print_function, division, absolute_import
import os
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import numpy as np
import argparse
import pandas as pd
from math import sqrt
from WDviewer import selectidx

#Dom Rowan REU 2018

desc="""
WDColorMag: Produce color mag plot with shading based on best rank. Used to help identify interesting sources/confirm ranking
"""

#Main Plotting function:
def main(pick, view, region, mark, markgroup):
    #Path assertions
    assert(os.path.isfile("Output/AllData.csv"))
    assert(os.path.isfile("/home/dmrowan/WhiteDwarfs/Catalogs/MainCatalog_reduced_simbad_asassn.csv"))

    #Read in dataframes
    df_rank = pd.read_csv("Output/AllData.csv")
    bigcatalog = pd.read_csv("/home/dmrowan/WhiteDwarfs/Catalogs/MainCatalog_reduced_simbad_asassn.csv")

    #Only looking at top 3000 sources
    df_rank_reduced = df_rank[:3001]
    #Read in ranks, sources
    no_gaia_data_counter = 0

    bp_rp_list = []
    absolute_mag_list = []
    rank_list = []
    source_list = []
    band_list = []
    for source, i in zip(df_rank_reduced['SourceName'], range(len(df_rank_reduced['SourceName']))):
        #Find source in BigCatalog to get gaia mag and color
        nhyphens = len(np.where(np.array(list(source)) == '-')[0])
        if source[0:4] == 'Gaia':
            bigcatalog_idx = np.where(bigcatalog['MainID'] == source.replace('-', ' '))[0]
        elif source[0:5] == 'ATLAS':
            bigcatalog_idx = np.where(bigcatalog['MainID'] == source)[0]
        elif source[0:2] == 'GJ':
            bigcatalog_idx = np.where(bigcatalog['MainID'] == source.replace('-', ' '))[0]
        elif source[0:2] == 'CL':
            bigcatalog_idx = np.where(bigcatalog['MainID'] == source.replace('-', ' '))[0]
        elif source[0:2] == 'LP':
            if nhyphens == 2:
                bigcatalog_idx = np.where(bigcatalog['MainID'] == source.replace('-', ' ', 1))[0]
            else:
                bigcatalog_idx = np.where(bigcatalog['MainID'] == source)[0]
        elif source[0:2] == 'V*':
            bigcatalog_idx = np.where(bigcatalog['MainID'] == source.replace('-', ' '))[0]
        elif source[0:3] == '2QZ':
            bigcatalog_idx = np.where(bigcatalog['MainID'] == source.replace('-', ' ', 1))[0]
        else:
            if nhyphens == 1:
                bigcatalog_idx = np.where(bigcatalog['MainID'] == source.replace('-', ' ' ))[0]
            else:
                bigcatalog_idx = np.where(bigcatalog['MainID'] == source.replace('-', ' ',nhyphens-1))[0]

        if len(bigcatalog_idx) == 0:
                print(source, "Not in catalog")
                no_gaia_data_counter += 1
        else:
            #Grab Gaia information 
            bigcatalog_idx = bigcatalog_idx[0]
        
            gaia_parallax = bigcatalog['gaia_parallax'][bigcatalog_idx]
            if str(gaia_parallax) == 'nan':
                no_gaia_data_counter += 1
            elif gaia_parallax <= 0:
                no_gaia_data_counter += 1
            else:
                #Calcualte bp-rp, absolute mag
                g_mag = bigcatalog['gaia_g_mean_mag'][bigcatalog_idx]
                bp_mag = bigcatalog['gaia_bp_mean_mag'][bigcatalog_idx]
                rp_mag = bigcatalog['gaia_rp_mean_mag'][bigcatalog_idx]

                absolute_mag_list.append(g_mag + 5*np.log10(gaia_parallax/100))
     
                bp_rp_list.append(bp_mag - rp_mag)

                rank_list.append(df_rank_reduced['BestRank'][i])
                source_list.append(source)
                band_list.append(df_rank_reduced['Band'][i])


    assert(len(bp_rp_list) == len(absolute_mag_list) == len(rank_list) == len(source_list))

    #Plotting params
    df_plot = pd.DataFrame({'source':source_list, 'band':band_list,'rank':rank_list, 'absolutemag':absolute_mag_list, 'bp_rp':bp_rp_list})
    df_plot = df_plot.sort_values(by=['rank'])

    #Different plotting parameters. Using markgroup for paper plot generation
    if markgroup:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        fig.tight_layout(rect=[.03, 0.03, 1, 1])
        fig.subplots_adjust(wspace=.125)
        ax1.scatter(df_plot['bp_rp'], df_plot['absolutemag'], c='gray', alpha=.5, label='_nolegend_')
        ax2.scatter(df_plot['bp_rp'], df_plot['absolutemag'], c='gray', alpha=.5, label='_nolegend_')
    else:
        fig, ax1 = plt.subplots(1,1, figsize=(12,4))
        ax1.scatter(df_plot['bp_rp'], df_plot['absolutemag'], c=df_plot['rank'], cmap='autumn_r')
        ax1.colorbar().set_label("Rank", fontsize=20)
        plt.title("Gaia CMD \n Sources not included: {}".format(no_gaia_data_counter), fontsize=20)

    ax1.set_ylim(ymin=15, ymax=5.75)
    ax1.set_xlim(xmin=-.85, xmax=.65)
    ax1.minorticks_on()
    ax1.yaxis.set_ticks_position('both')
    ax1.xaxis.set_ticks_position('both')
    ax1.tick_params(direction='in', which='both', labelsize=15)
    ax1.tick_params('both', length=8, width=1.8, which='major')
    ax1.tick_params('both', length=4, width=1, which='minor')
    ax1.set_xlabel("Gaia Color BP-RP", fontsize=15)
    ax1.set_ylabel("Absolute Gaia GMag", fontsize=15)
    
    #Second subplot params
    if markgroup:
        ax2.set_ylim(ymin=12.65, ymax=11.35)
        ax2.set_xlim(xmin=-.022, xmax=.16)
        ax2.minorticks_on()
        ax2.yaxis.set_ticks_position('both')
        ax2.xaxis.set_ticks_position('both')
        ax2.tick_params(direction='in', which='both', labelsize=15)
        ax2.tick_params('both', length=8, width=1.8, which='major')
        ax2.tick_params('both', length=4, width=1, which='minor')
        ax2.set_xlabel("Gaia Color BP-RP", fontsize=15)

    if mark is not None:
        #Determine if idx or source input:
        if mark.isdigit():
            mark = df_rank['SourceName'][int(mark)]

        if mark not in list(df_plot['source']):
            print("No gaia information on ", source)
        else:
            for i in range(len(df_plot['source'])):
                if df_plot['source'][i] == mark:
                    ii = i
                    break
            print(ii)
            print(df_plot.loc[ii])
            plt.plot([df_plot['bp_rp'][ii]], [df_plot['absolutemag'][ii]], 'o', color='green', ms=12)
        
        plt.show()

    if markgroup:
        assert(os.path.isfile("/home/dmrowan/WhiteDwarfs/InterestingSources/IS.csv"))
        df_IS = pd.read_csv("/home/dmrowan/WhiteDwarfs/InterestingSources/IS.csv")
        for idx in range(len(df_IS['MainID'])):

            #Define color and labelnum
            labelnum = df_IS['labelnum'][idx]
            if df_IS['type'][idx] == 'Pulsator':
                typecolor = 'red'
            elif df_IS['type'][idx] == 'KnownPulsator':
                typecolor = 'green'
            else:
                typecolor = 'blue'

            #Iterate through points, selecting subplot and marking based on type
            for i in range(len(df_plot['source'])):
                if df_plot['source'][i] == df_IS['MainID'][idx]:
                    print(df_plot.loc[i])
                    ax1.scatter(df_plot['bp_rp'][i], df_plot['absolutemag'][i], c=typecolor, s=100, label='_nolegend_')
                    if (df_plot['bp_rp'][i] > -.022 ) and (df_plot['bp_rp'][i] < .16) and (df_plot['absolutemag'][i] < 12.65) and (df_plot['absolutemag'][i] > 11.35):
                        ax2.scatter(df_plot['bp_rp'][i], df_plot['absolutemag'][i], c=typecolor, s=100, label='_nolegend_')
                        #Include anotations w/ num label
                        ax2.annotate(str(labelnum), (df_plot['bp_rp'][i], df_plot['absolutemag'][i]), fontsize=12.5)
                    else:
                        ax1.annotate(str(labelnum), (df_plot['bp_rp'][i], df_plot['absolutemag'][i]), fontsize=12.5)

        #Additional plotting params
        ax1.plot([-.022, -.022, .16, .16, -.022], [12.65, 11.35, 11.35, 12.65, 12.65], color='black', ls='--', lw=4)
        ax1.scatter(x=0, y=0, c='red', s=100, label='Pulsator')
        ax1.scatter(x=0, y=0, c='green', s=100, label='KnownPulastor')
        ax1.scatter(x=0, y=0, c='blue', s=100, label='Eclipse')
        ax1.legend(loc=3, fontsize=15, scatterpoints=1)
     
        #Artist connecting point for zoom window
        con1 = ConnectionPatch(xyA=(.16,11.35), xyB=(-.022, 11.35), coordsA="data", coordsB="data", axesA=ax1, axesB=ax2, color='black', ls='--')
        con2 = ConnectionPatch(xyA=(.16, 12.65), xyB=(-.022, 12.65), coordsA="data", coordsB="data", axesA=ax1, axesB=ax2, color='black', ls='--')
        ax1.add_artist(con1)
        ax1.add_artist(con2)

        fig.savefig("/home/dmrowan/WhiteDwarfs/InterestingSources/CMDplot.pdf")
    #Grab specific point
    elif pick:
        pick_point = plt.ginput(1, show_clicks=True)[0]
        coords = []
        for i in range(len(df_plot['absolutemag'])):
            color_coords = df_plot['bp_rp'][i]
            mag_coords = df_plot['absolutemag'][i]
            coords.append( (color_coords, mag_coords) )
        distances = []
        for i in range(len(coords)):
            deltax = pick_point[0] - coords[i][0]
            deltay = pick_point[1] - coords[i][1]
            distances.append(sqrt(deltax**2 + deltay**2))
        assert(len(distances) == len(coords))
        closest_indicies = np.where(np.array(distances) == np.nanmin(distances))[0]
        for idx in closest_indicies:
            if not view:
                idx_alldata = np.where( (df_rank['SourceName'] == df_plot['source'][idx]) & (df_rank['Band'] == df_plot['band'][idx]) )[0]
                idx_alldata = idx_alldata[0] + 2
                print(idx_alldata)
            print(df_plot.loc[idx])
        

        if view:
            for idx in closest_indicies:
                idx_alldata = np.where( (df_rank['SourceName'] == df_plot['source'][idx]) & (df_rank['Band'] == df_plot['band'][idx]) )[0]
                idx_alldata = idx_alldata[0] + 2
                print(idx_alldata)
                
                selectidx(idx_alldata, comment=False)

        plt.show()

    #Grab all points within a specified region
    elif region:
        #Draw box and blox corners/lines
        point_list = plt.ginput(2, show_clicks=True)
        xvals = [ tup[0] for tup in point_list ]
        yvals = [ tup[1] for tup in point_list ] 
        plt.plot([xvals[0], xvals[0]], [yvals[0], yvals[1]], 'b-')
        plt.plot([xvals[1], xvals[1]], [yvals[0], yvals[1]], 'b-')
        plt.plot([xvals[0], xvals[1]], [yvals[0], yvals[0]], 'b-')
        plt.plot([xvals[0], xvals[1]], [yvals[1], yvals[1]], 'b-')
        plt.plot(xvals, yvals, 'bo')
        

        df_region = pd.DataFrame({'source':[], 'band':[], 'rank':[], 'absolutemag':[], 'bp_rp':[]})

        for i in range(len(df_plot['absolutemag'])):
            color = df_plot['bp_rp'][i]
            mag = df_plot['absolutemag'][i]
            if (color > min(xvals)) and (color < max(xvals)):
                if (mag > min(yvals)) and (color < max(yvals)):
                    df_temp = df_plot.ix[i]
                    df_region = df_region.append(df_temp, ignore_index=True)

        df_region = df_region.sort_values(by=['rank'], ascending=False)
        print(df_region)
        plt.show()


    else:
        plt.show()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("--pick", help="Use Lasso to select points and output sourcename", default=False, action='store_true')
    parser.add_argument("--view", help="View pdf of selected source", default=False, action='store_true')
    parser.add_argument("--region", help="Draw a box and grab info on all sources in region", default=False, action='store_true')
    parser.add_argument("--mark", help="Input either the idx of a source (in AllData) or its actual name and mark it's position on the CMD", default=None, type=str)
    parser.add_argument("--markgroup", help="Using this to make plots for paper", default=False, action='store_true')
    args= parser.parse_args()

    main(args.pick, args.view, args.region, args.mark, args.markgroup)
