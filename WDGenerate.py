#!/usr/bin/env python
from __future__ import print_function, division
import os
import argparse
import WDAssign
import WDPlot_pulsator
import WDColorMag
import WDsigmamag
import WDmaghist
import WD_EB
from WD_EB import WDEB
#Dom Rowan REU 2018

desc="""
WDGenerate: Generate all plots, tables, for paper
"""

def latexcommands(printcommands=False):
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
    n_new_pulsators = "{:,}".format(len(os.listdir(
        "InterestingSources/Pulsator")))
    n_known_pulsators = "{:,}".format(len(os.listdir(
        "InterestingSources/KnownPulsator")))

    eclipse_command = "\\newcommand{\\neclipse}{"+n_eclipses+" }"
    new_pulsators_command = ("\\newcommand{\\nnewpulsator}{"
                             +n_new_pulsators+" }")
    known_pulsators_command = ("\\newcommand{\\nknownpulsators}{"
                               +n_known_pulsators+" }")

    #Write file
    with open('WDcommands.tex', 'w') as f:
        f.write(survey_command + "\n")
        f.write(init_survey_command + "\n")
        f.write(eclipse_command + "\n")
        f.write(new_pulsators_command + "\n")
        f.write(known_pulsators_command + "\n")

    if printcommands:
        print(survey_command)
        print(init_survey_command)
        print(eclipse_command)
        print(new_pulsators_command)
        print(known_pulsators_command)


def main(eclipsegenerate, pulsatorgenerate):
    #Path assertions 
    assert(os.getcwd() == '/home/dmrowan/WhiteDwarfs')
    assert(os.path.isdir('InterestingSources'))
    assert(os.path.isdir('GalexData_run6'))

    #Run WDAssign
    os.chdir('InterestingSources')
    WDAssign.main(False)
    WDAssign.latextable()
    WDAssign.latex_known()

    #Run WD_EB
    WD_EB.main(iterations=1000)

    #Run WDPlot_pulsator.py
    WDPlot_pulsator.main(pulsatorgenerate)

    #Run WDmaghist.py
    WDmaghist.main()
    
    #Run WDColorMag.py
    os.chdir('../GalexData_run6')
    WDColorMag.main(False, False, False,None, True, False, True)
    WDColorMag.main(False, False, False, None, True, False, False)

    #Run WDsigmamag.py
    WDsigmamag.percentile(True)

    #Run WDlatexcommands.py
    os.chdir('../')
    assert(os.getcwd() == '/home/dmrowan/WhiteDwarfs')
    latexcommands
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
            "--generate_eclipse_info", 
            help="Generate eclipse information for WDPlot_eclipse.py", 
            default=False, action='store_true')
    parser.add_argument(
            "--generate_pulsator_info", 
            help="Generate pulsator information for WD_plot_pulsator.py",
            default=False, action='store_true')
    args= parser.parse_args()

    main(eclipsegenerate=args.generate_eclipse_info,
         pulsatorgenerate=args.generate_pulsator_info,
    )
