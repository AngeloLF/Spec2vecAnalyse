import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import os, sys, pickle, json, shutil
import coloralf as c

from types import SimpleNamespace
from time import time
from tqdm import tqdm

sys.path.append('./Spec2vecModels/')
from get_argv import get_argv

sys.path.append(f"./SpecSimulator")
from simulator import SpecSimulator
import utils_spec.psf_func as pf



if __name__ == "__main__":


    """

    results |-- output_simu |-- Args.train --- hist_params.json
            |               |-- Args.test  |-- <Folds.pred_folder>
            |                              |-- spectrum
            |                              |-- hist_params.json
            |                              |-- variable_params.pck
            |
            |
            |-- analyse     --- Args.score --- <Folds.pred_folder> --- Args.test (-> Paths.save)

                                                
    """

    tradargs = {
        "arg.0.0" : f"PSF Moffat $\gamma$",
    }


    total_time = time()

    Args = get_argv(sys.argv[1:], prog="analyse")

    Paths = SimpleNamespace()
    Folds = SimpleNamespace()


    # Define path test
    if "output" in Args.test : Paths.for_test = "./results"
    else : Paths.for_test = "./results/output_simu"
    Paths.test = f"{Paths.for_test}/{Args.test}"

    Paths.for_train = './results/output_simu'
    Paths.train = f"{Paths.for_train}/{Args.train}"

    Paths.analyse = f"./results/analyse"
    Paths.score = f"{Paths.analyse}/{Args.score}"

    Folds.pred_folder = f"pred_{Args.fullname}"
    Paths.pred_folder = f"{Paths.score}/{Folds.pred_folder}"

    os.makedirs(Paths.analyse, exist_ok=True)
    os.makedirs(Paths.score, exist_ok=True)
    os.makedirs(Paths.pred_folder, exist_ok=True)
    

    # chargement des params du train
    try:
        with open(f"{Paths.train}/hist_params.json", 'r') as f:
            train_params = json.load(f)
    except:
        train_params = dict()


    # For each folder test (like output construct with lsp)
    Paths.save = f"{Paths.pred_folder}/{Args.test}"
    if Args.test in os.listdir(Paths.pred_folder) : shutil.rmtree(Paths.save)
    os.mkdir(f"{Paths.save}")
    # os.mkdir(f"{Paths.save}")



    trues = sorted(os.listdir(f"{Paths.test}/opa"))
    preds = sorted(os.listdir(f"{Paths.test}/{Folds.pred_folder}"))
    if len(trues) != len(preds) : raise Exception(f"In [analyse_FOPA.py], trues and preds don't have the same size ({len(trues)} and {len(preds)})")
    n = len(trues)

    ot, pt, at = np.zeros(n), np.zeros(n), np.zeros(n)
    op, pp, ap = np.zeros(n), np.zeros(n), np.zeros(n)

    for i, file in enumerate(trues):

        ti = np.load(f"{Paths.test}/opa/{file}")
        pi = np.load(f"{Paths.test}/{Folds.pred_folder}/{file}")

        if "FOPA" in Args.model:

            ot[i], pt[i], at[i] = ti
            op[i], pp[i], ap[i] = pi

        elif "FOBIQ" in Args.model:

            ot[i], pt[i], at[i] = ti
            op[i] = pi[0]



    resume = dict()

    for i, (atmop, vt, vp) in enumerate([("Ozone", ot, op), ("PWV", pt, pp), ("Aerosols", at, ap)]):


        rangeParam = train_params[f"ATM_{atmop.upper()}"]
        asort = np.argsort(vt)
        diff = vt - vp


        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 6), gridspec_kw={'height_ratios': [3, 1]})
        # plt.suptitle(f"Prediction of {atmop}")

        ax1.axhline(rangeParam[0], color='green', linestyle=':')
        ax1.axhline(rangeParam[1], color='green', linestyle=':')

        ax1.plot(vt[asort], color='g', label="True")
        ax1.plot(vp[asort], color='r', label="Pred")

        ax1.set_ylabel(atmop)
        ax1.legend()

        ax2.plot(diff[asort], color='k')
        ax2.set_ylabel("Residus")
        ax2.set_xlabel("Num test")

        plt.savefig(f"{Paths.save}/num_{atmop}.png")
        plt.close()




        plt.figure(figsize=(12, 8))
    
        sco = np.sum(np.abs(diff)) / n
        std = np.std(np.abs(diff))

        plt.hist(diff, color='r', edgecolor='k', bins=50)
        plt.title(f"Hist of {atmop} error : {sco:.4f}")

        plt.savefig(f"{Paths.save}/hist_{atmop}.png")
        plt.close()

        resume[atmop.lower()] = [sco, std]



        with open(f"{Paths.save}/resume.json", 'w') as f:
            json.dump(resume, f, indent=4)










