#!/usr/bin/env python
from __future__ import print_function, division
import os
import argparse
import pandas as pd
import numpy as np
#Dom Rowan REU 2018

desc="""
WDAssign: Iterate through interesting sources and generate CSV for use in plot creation. Also assign number for use in labeling.
"""

def main():
    #Path assertions 
    assert(os.getcwd() == '/home/dmrowan/WhiteDwarfs/InterestingSources')
    assert(os.path.isdir("KnownPulsator"))
    assert(os.path.isdir("Pulsator"))
    assert(os.path.isdir("Eclipse"))
    bigcatalog_path = "/home/dmrowan/WhiteDwarfs/Catalogs/MainCatalog_reduced_simbad_asassn.csv"
    assert(os.path.isfile(bigcatalog_path))
    sigmamagpath = "/home/dmrowan/WhiteDwarfs/GalexData_run5/Catalog/SigmaMag_reduced.csv"
    assert(os.path.isfile(sigmamagpath))
    percentilepath = "/home/dmrowan/WhiteDwarfs/GalexData_run5/Catalog/magpercentiles.csv"
    assert(os.path.isfile(percentilepath))

    #Load in source names
    knownpulsators = [ dirname for dirname in os.listdir("KnownPulsator") ]
    newpulsators = [ dirname for dirname in os.listdir("Pulsator") ] 
    eclipses = [ dirname for dirname in os.listdir("Eclipse") ] 

    interestingsources = knownpulsators + newpulsators + eclipses
    objecttypes = ["KnownPulsator"]*len(knownpulsators) + ["Pulsator"]*len(newpulsators) + ["Eclipse"]*len(eclipses)

    #Need to query big catalog for g mag, ra, dec
    bigcatalog = pd.read_csv(bigcatalog_path)
    g_list = []
    ra_list = []
    dec_list = []
    for name in interestingsources:
        nhyphens = len(np.where(np.array(list(name)) == '-')[0])
        if name[0:4] == 'Gaia':
            bigcatalog_idx = np.where(bigcatalog['MainID'] == name.replace('-', ' '))[0]
        elif name[0:5] == 'ATLAS':
            bigcatalog_idx = np.where(bigcatalog['MainID'] == name)[0]
        elif name[0:2] == 'GJ':
            bigcatalog_idx = np.where(bigcatalog['MainID'] == name.replace('-', ' '))[0]
        elif name[0:2] == 'CL':
            bigcatalog_idx = np.where(bigcatalog['MainID'] == name.replace('-', ' '))[0]
        elif name[0:2] == 'LP':
            if nhyphens == 2:
                bigcatalog_idx = np.where(bigcatalog['MainID'] == name.replace('-', ' ', 1))[0]
            else:
                bigcatalog_idx = np.where(bigcatalog['MainID'] == name)[0]
        elif name[0:2] == 'V*':
            bigcatalog_idx = np.where(bigcatalog['MainID'] == name.replace('-', ' '))[0]
        elif name[0:3] == '2QZ':
            bigcatalog_idx = np.where(bigcatalog['MainID'] == name.replace('-', ' ', 1))[0]
        else:
            if nhyphens == 1:
                bigcatalog_idx = np.where(bigcatalog['MainID'] == name.replace('-', ' ' ))[0]
            else:
                bigcatalog_idx = np.where(bigcatalog['MainID'] == name.replace('-', ' ',nhyphens-1))[0]

        if len(bigcatalog_idx) == 0:
            print(name, "Not in catalog")
            g_list.append("")
            ra_list.append("")
            dec_list.append("")
        else:
            bigcatalog_idx = bigcatalog_idx[0]
            g_list.append(bigcatalog['gaia_g_mean_mag'][bigcatalog_idx])
            ra_list.append(bigcatalog['ra'][bigcatalog_idx])
            dec_list.append(bigcatalog['dec'][bigcatalog_idx])

    #Query sigmamag csv's go get sigma m, m_ab, and our variability metric
    metric= []
    sigma_list = []
    m_ab_list = []
    df_sm = pd.read_csv(sigmamagpath)
    percentile_df = pd.read_csv(percentilepath)
    magbins = percentile_df['magbin']
    magbins = np.array(magbins)
    percentile50 = percentile_df['median']
    for name in interestingsources:
        idx = np.where(df_sm['Source'] == name)[0]
        #Cases for no idx, one idx, two idx
        if len(idx) == 0:
            print(name, "Not in sigma mag")
            metric.append("")
        elif len(idx) == 1:
            idx = idx[0]
            m_ab = df_sm['m_ab'][idx]
            sigma_m = df_sm['sigma_m'][idx]
            sigmamag_idx = np.where(abs(m_ab-magbins) == min(abs(m_ab-magbins)))[0]
            sigmafit_val = float(percentile50[sigmamag_idx])
            metric.append(sigma_m / sigmafit_val)
            m_ab_list.append(m_ab)
            sigma_list.append(sigma_m)
        else:
            metricsum = 0
            m_ab_sum = 0
            sigma_sum = 0
            for i in idx:
                m_ab = df_sm['m_ab'][i]
                sigma_m = df_sm['sigma_m'][i]
                sigmamag_idx = np.where(abs(m_ab-magbins) == min(abs(m_ab-magbins)))[0]
                sigmafit_val = float(percentile50[sigmamag_idx])
                metricsum+= (sigma_m / sigmafit_val)
                m_ab_sum += m_ab
                sigma_sum += sigma_m
            metric.append(metricsum / len(idx))
            m_ab_list.append(m_ab_sum / len(idx))
            sigma_list.append(sigma_sum / len(idx))

    #Create df, sort values by ra, assign label number for use in plots
    df_output = pd.DataFrame({"MainID":interestingsources, "ra":ra_list, "dec":dec_list, "g":g_list, "type":objecttypes, "metric":metric, "m_ab":m_ab_list, "sigma_m":sigma_list})
    df_output = df_output.sort_values(by=["ra"])
    df_output['labelnum'] = list(range(len(interestingsources)))
    df_output.to_csv("IS.csv", index=False)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=desc)
    args= parser.parse_args()

    main()
