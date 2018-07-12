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

def main(ppuls):
    #Path assertions 
    assert(os.getcwd() == '/home/dmrowan/WhiteDwarfs/InterestingSources')
    assert(os.path.isdir("KnownPulsator"))
    assert(os.path.isdir("Pulsator"))
    assert(os.path.isdir("Eclipse"))
    assert(os.path.isdir("PossiblePulsator"))
    bigcatalog_path = "/home/dmrowan/WhiteDwarfs/Catalogs/MainCatalog_reduced_simbad_asassn.csv"
    assert(os.path.isfile(bigcatalog_path))
    sigmamagpath_NUV = "/home/dmrowan/WhiteDwarfs/GalexData_run5/Catalog/SigmaMag_NUV.csv"
    sigmamagpath_FUV = "/home/dmrowan/WhiteDwarfs/GalexData_run5/Catalog/SigmaMag_FUV.csv"
    assert(os.path.isfile(sigmamagpath_NUV))
    assert(os.path.isfile(sigmamagpath_FUV))
    percentilepath_NUV = "/home/dmrowan/WhiteDwarfs/GalexData_run5/Catalog/magpercentiles_NUV.csv"
    percentilepath_FUV = "/home/dmrowan/WhiteDwarfs/GalexData_run5/Catalog/magpercentiles_FUV.csv"
    assert(os.path.isfile(percentilepath_NUV))
    assert(os.path.isfile(percentilepath_FUV))

    #Load in source names
    knownpulsators = [ dirname for dirname in os.listdir("KnownPulsator") ]
    newpulsators = [ dirname for dirname in os.listdir("Pulsator") ] 
    eclipses = [ dirname for dirname in os.listdir("Eclipse") ] 

    if ppuls:
        possiblepulsators = [ dirname for dirname in os.listdir("PossiblePulsator") ] 
        objecttypes = ["Possible"] * len(possiblepulsators)
        interestingsources = possiblepulsators
    else:
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

    #Query sigmamag csv's go get sigma m, m_ab, and our variability metric NUV
    metric_NUV= []
    sigma_list_NUV = []
    m_ab_list_NUV = []
    df_sm_NUV = pd.read_csv(sigmamagpath_NUV)
    percentile_df_NUV = pd.read_csv(percentilepath_NUV)
    magbins_NUV = percentile_df_NUV['magbin']
    magbins_NUV = np.array(magbins_NUV)
    percentile50_NUV = percentile_df_NUV['median']
    for name in interestingsources:
        idx = np.where(df_sm_NUV['Source'] == name)[0]
        #Cases for no idx, one idx, two idx
        if len(idx) == 0:
            print(name, "NUV not in sigma mag")
            metric_NUV.append("")
            sigma_list_NUV.append("")
            m_ab_list_NUV.append("")
        elif len(idx) == 1:
            idx = idx[0]
            m_ab = df_sm_NUV['m_ab'][idx]
            sigma_m = df_sm_NUV['sigma_m'][idx]
            sigmamag_idx = np.where(abs(m_ab-magbins_NUV) == min(abs(m_ab-magbins_NUV)))[0]
            sigmafit_val = float(percentile50_NUV[sigmamag_idx])
            metric_NUV.append(sigma_m / sigmafit_val)
            m_ab_list_NUV.append(m_ab)
            sigma_list_NUV.append(sigma_m)

    #Query sigmamag csv's go get sigma m, m_ab, and our variability metric FUV
    metric_FUV= []
    sigma_list_FUV = []
    m_ab_list_FUV = []
    df_sm_FUV = pd.read_csv(sigmamagpath_FUV)
    percentile_df_FUV = pd.read_csv(percentilepath_FUV)
    magbins_FUV = percentile_df_FUV['magbin']
    magbins_FUV = np.array(magbins_FUV)
    percentile50_FUV = percentile_df_FUV['median']
    for name in interestingsources:
        idx = np.where(df_sm_FUV['Source'] == name)[0]
        #Cases for no idx, one idx, two idx
        if len(idx) == 0:
            print(name, "FUV not in sigma mag")
            metric_FUV.append("")
            sigma_list_FUV.append("")
            m_ab_list_FUV.append("")
        elif len(idx) == 1:
            idx = idx[0]
            m_ab = df_sm_FUV['m_ab'][idx]
            sigma_m = df_sm_FUV['sigma_m'][idx]
            sigmamag_idx = np.where(abs(m_ab-magbins_FUV) == min(abs(m_ab-magbins_FUV)))[0]
            sigmafit_val = float(percentile50_FUV[sigmamag_idx])
            metric_FUV.append(sigma_m / sigmafit_val)
            m_ab_list_FUV.append(m_ab)
            sigma_list_FUV.append(sigma_m)

    #Create df, sort values by ra, assign label number for use in plots
    df_output = pd.DataFrame({
        "MainID":interestingsources,
        "ra":ra_list, 
        "dec":dec_list, 
        "g":g_list, 
        "type":objecttypes,
        "metric_NUV":metric_NUV, 
        "m_ab_NUV":m_ab_list_NUV, 
        "sigma_m_NUV":sigma_list_NUV,
        "metric_FUV":metric_FUV, 
        "m_ab_FUV":m_ab_list_FUV, 
        "sigma_m_FUV":sigma_list_FUV})

    df_output = df_output.sort_values(by=["ra"])
    df_output['labelnum'] = list(range(len(interestingsources)))
    if ppuls:
        df_output.to_csv("IS_possible.csv", index=False)
    else:
        df_output.to_csv("IS.csv", index=False)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("--ppuls", help="Create IS_possible.csv which contains information on possible pulsators", default=False, action='store_true')
    args= parser.parse_args()

    main(ppuls=args.ppuls)
