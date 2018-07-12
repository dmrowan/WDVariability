#!/usr/bin/env python
from __future__ import print_function, division
import os
import argparse
import subprocess
import pandas as pd
#Dom Rowan, REU 2018
desc = """
LatexCommands.py: Creates a file WDcommands.tex that contains commands for main.tex file
"""

def main():
    if os.path.isfile("WDcommands.tex"):
        print("Overwriting current WDcommands.tex file")
        subprocess.run(['rm', 'WDcommands.tex'])

    #Survey size
    df_catalog = pd.read_csv("Catalogs/MainCatalog_reduced_simbad_asassn.csv")
    n_survey = "{:,}".format(len(list(set(list(df_catalog['MainID'])))))
    survey_command = "\\newcommand{\\surveysize}{"+n_survey+" }"

    #Initial catalog size
    df_ic = pd.read_csv("Catalogs/Allsources_allexpt.csv")
    MainID_list = []
    for i in range(len(df_ic)):
        if str(df_ic['MWDDDesignation'][i]) != 'nan':
            MainID_list.append(df_ic['MWDDDesignation'][i])
        elif str(df_ic['ATLASDesignation'][i]) != 'nan':
            MainID_list.append(df_ic['ATLASDesignation'][i])
        elif str(df_ic['SDSSDesignation'][i]) != 'nan':
            MainID_list.append(df_ic['SDSSDesignation'][i])
        else:
            MainID_list.append(df_ic['GaiaDesignation'][i])

    n_initial = "{:,}".format(len(list(set(list(MainID_list)))))
    init_survey_command = "\\newcommand{\\initsize}{"+n_initial+" }"

    #Interesting source numbers
    n_eclipses = "{:,}".format(len(os.listdir("InterestingSources/Eclipse")))
    n_new_pulsators = "{:,}".format(len(os.listdir("InterestingSources/Pulsator")))
    n_known_pulsators = "{:,}".format(len(os.listdir("InterestingSources/KnownPulsator")))

    eclipse_command = "\\newcommand{\\neclipse}{"+n_eclipses+" }"
    new_pulsators_command = "\\newcommand{\\nnewpulsator}{"+n_new_pulsators+" }"
    known_pulsators_command = "\\newcommand{\\nknownpulsators}{"+n_known_pulsators+" }"

    with open('WDcommands.tex', 'w') as f:
        f.write(survey_command + "\n")
        f.write(init_survey_command + "\n")
        f.write(eclipse_command + "\n")
        f.write(new_pulsators_command + "\n")
        f.write(known_pulsators_command + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=desc)
    args= parser.parse_args()

    main()


