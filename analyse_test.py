import numpy as np
import matplotlib.pyplot as plt

import os, sys, pickle, json, shutil
import coloralf as c

from types import SimpleNamespace
from time import time
from tqdm import tqdm

sys.path.append('./Spec2vecModels/')
from get_argv import get_argv




def compute_score_L1(true, pred, give_norm_array=False):

    t = np.copy(true)
    p = np.copy(pred)

    p[p < 0] = 0
    t[t < 0] = 0

    # non norma
    max_pt = np.max(np.array([t, p]), axis=0)
    diff = np.abs(t-p)
    score = np.sum(diff) / np.sum(t)

    # norma
    tn = t / np.max(t)
    pn = p / np.max(p)
    max_pt_norma = np.max(np.array([tn, pn]), axis=0)
    diff_norma = np.abs(tn-pn)
    score_norma = np.sum(diff_norma) / np.sum(tn)

    if give_norm_array : return tn, pn, score, score_norma
    else : return score, score_norma



def compute_score_chi2(true, pred, Csigma_chi2=12, norma=800, give_norm_array=False):

    t = np.copy(true)
    p = np.copy(pred)

    p[p < 0] = 0
    t[t < 0] = 0

    # non norma
    chi2 = np.sum((true - pred)**2 / (true + Csigma_chi2**2)) / norma

    # norma
    norma = np.sum((true - pred)**2 / (true + 1.**2)) / norma

    if give_norm_array : return true, pred, chi2, norma
    else : return chi2, norma



def compute_score(name, true, pred, give_norm_array=False):

    if   name == "L1"   : return compute_score_L1(true, pred, give_norm_array=give_norm_array)
    elif name == "chi2" : return compute_score_chi2(true, pred, give_norm_array=give_norm_array)
    else : raise Exception(f"Unknow score name {name} in analyse_test.compute_score.")



def makeOneSpec(Args, Paths, Folds, res, varp, n, give_norma, give_image, savename, gain=3.):

    with open(f"{Paths.test}/hparams.json", 'r') as f:
        hparams = json.load(f)

    x = Args.wl # np.arange(hparams["LAMBDA_MIN"], hparams["LAMBDA_MAX"], hparams["LAMBDA_STEP"])
    num_spec = res["num"][n]

    pred = np.load(f"{Paths.test}/{Folds.pred_folder}/{Args.folder_output}_{num_spec}.npy")
    true = np.load(f"{Paths.test}/{Args.folder_output}/{Args.folder_output}_{num_spec}.npy")
    image = np.load(f"{Paths.test}/image/image_{num_spec}.npy")

    tn, pn, s, sn = compute_score(Args.score, true, pred, give_norm_array=True)
    if give_norma : true, pred = tn, pn

    plt.figure(figsize=(16, 8))
    if give_image : plt.subplot(211)

    plt.plot(x, true, c='g', label='True')
    plt.plot(x, pred, c='r', label='Pred')

    plt.title(f"For {Args.model_loss} train with {Args.fulltrain_str}_{Args.lr_str} : {s*100:.1f} % [non norma] | {sn*100:.1f} % [norma] | Flux : {res['flux'][n]/gain/1000:.0f} kADU")
    plt.scatter([], [], marker='d', label=f"Target : {varp['TARGET'][n]}", color='k')
    for key, val in varp.items():
        if key != "TARGET": 
            skey = key if key not in tradargs.keys() else tradargs[key]
            plt.scatter([], [], marker='*', label=f"{skey} = {val[n]:.2f}", color='k')
    plt.legend() 
    plt.xlabel(f"$\lambda$ (nm)")
    plt.ylabel(f"{Paths.test}/*/{Args.folder_output}_{num_spec}.npy")
    
    if give_image:
        plt.subplot(212)
        plt.imshow(np.log10(image+1), cmap='gray')
        plt.savefig(f"{Paths.save}/example_image/{savename}.png")
        plt.close()
    else:    
        plt.savefig(f"{Paths.save}/example_spectrum/{savename}.png")
        plt.close()



def makeResidus(Args, Paths, Folds, res, n, savename, C=12., gain=3.):

    with open(f"{Paths.test}/variable_params.pck", "rb") as f:
        varp = pickle.load(f)

    num_spec = res["num"][n]
    n = int(num_spec)


    x = Args.wl

    yt = np.load(f"{Paths.test}/{Args.folder_output}/{Args.folder_output}_{num_spec}.npy")
    yp = np.load(f"{Paths.test}/{Folds.pred_folder}/{Args.folder_output}_{num_spec}.npy")

    residu = (yt - yp) 
    nor = np.sqrt(yt + C**2)

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(20, 8), gridspec_kw={'height_ratios': [3, 1]})

    ax1.set_title(f"For {Args.model_loss} train with {Args.fulltrain_str}_{Args.lr_str} | Flux : {np.sum(yt)/gain/1000:.0f} kADU")

    ax1.plot(x, yt, label='Spectrum to predict', color="g")
    ax1.plot(x, yp, label='Model prediction', color="r")

    ax1.scatter([], [], marker='d', label=f"Target : {varp['TARGET'][n]}", color='k')
    for key, val in varp.items():
        if key != "TARGET": 
            skey = key if key not in tradargs.keys() else tradargs[key]
            ax1.scatter([], [], marker='*', label=f"{skey} = {val[n]:.2f}", color='k')

    ax1.set_ylabel(f"{Args.test}/*/spectrum_{num_spec}.npy")
    ax1.legend()

    ax2.axhline(0, color='k', linestyle='--', linewidth=1)
    ax2.errorbar(x, residu / nor, yerr=1, marker='.', linestyle='', color='k', linewidth=0.5)
    ax2.set_xlabel(f"$\lambda$ (nm)")
    ax2.set_ylabel(f"$\chi^2$")

    plt.tight_layout()
    plt.savefig(f"{Paths.save}/residus/{savename}_residus.png")
    plt.close()


    yts = np.sort(yt)
    dmax = int(np.max(yts[1:] - yts[:-1])*2) + 1
    nbins = int(np.max(yt) / dmax)

    xbin = np.zeros(nbins)
    resb = np.zeros(nbins)
    ress = np.zeros(nbins)

    for i in range(nbins):

        resi = residu[(yt > i*dmax) & (yt < (i+1)*dmax)]

        xbin[i] = dmax * (0.5 + i)
        resb[i] = np.mean(resi**2)
        ress[i] = np.std(resi**2)


    plt.figure(figsize=(16, 10))
    plt.plot(yt, residu**2, '.k', alpha=0.5)
    plt.errorbar(xbin, resb, yerr=ress, color="r", linestyle="", marker=".")
    plt.xlabel(r"$y_{true}$")
    plt.ylabel(f"$res^2$")
    plt.savefig(f"{Paths.save}/residus/{savename}_res2.png")
    plt.close()





def open_fold(args, paths, folds, nb_level=10):

    """
    dict res : 
        * keys `classic` & `norma` give a matrice (n_target, n_var)
        * axis 0 : len of n_target
        * axis 1 : len of n_var
    """

    files = os.listdir(f"{paths.test}/{Args.folder_output}")

    with open(f"{paths.test}/hist_params.json", 'r') as f:
        params = json.load(f)

    with open(f"{paths.test}/variable_params.pck", "rb") as f:
        var_params = pickle.load(f)

    res = {"classic" : np.zeros(params["nb_simu"]),
           "norma"   : np.zeros(params["nb_simu"]),
           "flux"    : np.zeros(params["nb_simu"]),
           "file"    : np.zeros(params["nb_simu"]).astype(str),
           "num"     : np.zeros(params["nb_simu"]).astype(str)}

    # loading of each spectrum
    for i, file in enumerate(files):

        num_spec_str = file.split("_")[-1][:-4]

        pred = np.load(f"{paths.test}/{folds.pred_folder}/{file}")
        true = np.load(f"{paths.test}/{Args.folder_output}/{file}")

        res["flux"][i] = np.sum(true) # flux in adu ~

        score, score_norma = compute_score(Args.score, true, pred)
        res["classic"][i] = score
        res["norma"][i] = score_norma
        res["file"][i] = file
        res["num"][i] = num_spec_str

    plt.figure(figsize=(12, 8))
    for mode, col in [("classic", "g"), ("norma", "r")]:
        
        plt.plot(res["flux"], res[mode]*100, color=col, label=mode, linestyle="", marker=".", alpha=0.7)
    plt.xlabel("Flux (en ADU)")
    plt.ylabel(f"Score (%)")
    plt.xscale("log")
    plt.legend()
    plt.savefig(f"{Paths.save}/flux2score.png")
    plt.close()



    for mode in ["classic", "norma"]:
        isNorma = True if mode == "norma" else False

        for i, level in enumerate(np.linspace(np.min(res[mode]), np.max(res[mode]), nb_level)):

            near = np.argmin(np.abs(res[mode]-level))
            makeOneSpec(args, paths, folds, res, var_params, near, give_norma=isNorma, give_image=False, savename=f"Level{i}_{mode}")
            makeOneSpec(args, paths, folds, res, var_params, near, give_norma=isNorma, give_image=True, savename=f"Level{i}_{mode}")
            if mode == "classic" : makeResidus(args, paths, folds, res, near, savename=f"Level{i}_{mode}")




    return res, var_params




def showTrainParams(train_params, var_name):
    """
    Pour mettre des axvline pour renseigner de la plage d'entrainement du modèle
    """

    if var_name in train_params.keys():
        minTrain, maxTrain = train_params[var_name]
        plt.axvline(minTrain, color='k', linestyle=':', label='In training')
        plt.axvline(maxTrain, color='k', linestyle=':')



def plotMinMaxScore(X, Y, minXY, maxXY, dX):

    plt.scatter(X[0] + minXY[0] * dX + dX/2, Y[minXY[1]], marker="d", facecolor='white', color='k')
    plt.scatter(X[0] + maxXY[0] * dX + dX/2, Y[maxXY[1]], marker="d", facecolor='darkred', color='k')



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
    with open(f"{Paths.train}/hist_params.json", 'r') as f:
        train_params = json.load(f)



    # For each folder test (like output construct with lsp)
    Paths.save = f"{Paths.pred_folder}/{Args.test}"
    if Args.test in os.listdir(Paths.pred_folder) : shutil.rmtree(Paths.save)
    os.mkdir(f"{Paths.save}")
    os.mkdir(f"{Paths.save}/metric")
    os.mkdir(f"{Paths.save}/example_spectrum")
    os.mkdir(f"{Paths.save}/example_image")
    os.mkdir(f"{Paths.save}/residus")



    if Args.test == "output_test":

        sub_folds = [f"{path_test}/{Args.test}/{sf}" for sf in os.listdir(f"{path_test}/{Args.test}") if "test" in sf]
        nsub = len(sub_folds)

        all_res = {"classic":None, "norma":None}

        for j, sub_fold in enumerate(sub_folds):

            print(f"{c.lg}Make {sub_fold} [{j+1}/{nsub}] ...{c.d}")
            new_res, targets_labels = open_fold(fold=sub_fold, pred_folder=pred_folder, path4save=path4save, train_params=train_params, 
                select_model=select_model, select_train=select_train, path_train=path_train, score_type=select_score_type)

            if j == 0: 
                all_res = new_res
            else:
                for mode in all_res.keys():
                    all_res[mode] = new_res[mode] if all_res[mode] is None else np.hstack((all_res[mode], new_res[mode]))


        # Score for each targets
        plt.figure(figsize=(16, 8))
        x_positions = np.arange(len(targets_labels))
        for mode, col, dx in [("classic", "g", -0.15), ("norma", "r", 0.15)]:
            targets_mean = np.mean(all_res[mode], axis=1)
            targets_std = np.std(all_res[mode], axis=1)

            targets_min = np.min(all_res[mode], axis=1)
            targets_max = np.max(all_res[mode], axis=1)

            ymin = (targets_mean + targets_std) - targets_min
            ymax = targets_max - (targets_mean + targets_std)
            ymax[ymax < 0] = 0.0
            
            plt.bar(x_positions+dx, 2*targets_std, bottom=targets_mean-targets_std, yerr=[ymin, ymax], capsize=5, alpha=0.7, color=col, width=0.3)
            plt.errorbar(x_positions+dx, targets_mean, color='k', linestyle='', marker='.')


        plt.xlabel("Targets")
        plt.ylabel("Scores")
        plt.title(f"Score for each traget in {Args.test}")
        plt.xticks(x_positions, targets_labels, rotation=90)
        plt.grid(axis='y', linestyle='--')
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(f"{path4save}/for_target.png")
        plt.close()
        # plt.show()

        with open(f"{path4save}/resume.txt", "w") as f:

            for mode in ["classic", "norma"]:

                f.write(f"For the `{mode}` calcul : {np.mean(all_res[mode])*100:.2f} ~ {np.std(all_res[mode])*100:.2f} %\n")





    else:

        t0 = time()
        res, var = open_fold(Args, Paths, Folds)
        print(time()-t0)



        for key, val in var.items():

            if key != "TARGET":

                yscore = np.copy(res["classic"])
                ylabel = f"Score {Args.score}"

                if Args.score in ["L1"]: 
                    yscore *= 100
                    ylabel += " (%)"

                argsort = np.argsort(val)
                xs = val[argsort]
                ys = yscore[argsort]
                dmax = np.max(xs[1:] - xs[:-1]) * 1.01
                nbins = int((xs[-1] - xs[0]) / dmax) + 1

                if nbins > 50 or "test50bins" in sys.argv:
                    nbins = 50
                    dmax = (xs[-1] - xs[0]) / nbins

                x0 = xs[0]

                print(f"\nkey {key}")
                print(xs)
                print(dmax, nbins)

                xbin = np.zeros(nbins)
                ybin = np.zeros(nbins)
                ystd = np.zeros(nbins)

                for i in range(nbins):

                    resi = yscore[(val > x0 + i*dmax) & (val < x0 + (i+1)*dmax)]

                    xbin[i] = x0 + dmax * (0.5 + i)
                    ybin[i] = np.mean(resi)
                    ystd[i] = np.std(resi)


                print(xbin)
                print(ybin)

                plt.figure(figsize=(16, 10))
                plt.scatter(val, yscore, color='k', marker='+', alpha=0.5)
                plt.errorbar(xbin, ybin, yerr=ystd, color="r", linestyle="", marker=".")
                # plt.legend()
                plt.xlabel(f"Variable {key}")
                plt.ylabel(ylabel)

                if not Args.show:
                    plt.savefig(f"{Paths.save}/metric/{key}.png")
                    plt.close()
                else:
                    plt.show()

                    plt.plot(val, yscore, '+k-')
                    plt.plot(xs, ys, "-r.")
                    plt.show()
                    sys.exit()

            else:

                targets_labels = list(set(val))
                ntarget = len(targets_labels)

                targets_mean = {"norma":np.zeros(ntarget), "classic":np.zeros(ntarget)}
                targets_std = {"norma":np.zeros(ntarget), "classic":np.zeros(ntarget)}
                targets_min = {"norma":np.zeros(ntarget), "classic":np.zeros(ntarget)}
                targets_max = {"norma":np.zeros(ntarget), "classic":np.zeros(ntarget)}

                for i, target in enumerate(targets_labels):

                    for mode in ["norma", "classic"]:

                        y = res[mode][val == target]
                        targets_mean[mode][i] = np.mean(y)
                        targets_std[mode][i] = np.std(y)
                        targets_min[mode][i] = np.min(y)
                        targets_max[mode][i] = np.max(y)

                # Score for each targets
                plt.figure(figsize=(16, 8))
                x_positions = np.arange(len(targets_labels))
                for mode, col, dx in [("classic", "g", -0.15), ("norma", "r", 0.15)]:

                    ymin = (targets_mean[mode] + targets_std[mode]) - targets_min[mode]
                    ymax = targets_max[mode] - (targets_mean[mode] + targets_std[mode])
                    ymax[ymax < 0] = 0.0
                    
                    plt.bar(x_positions+dx, 2*targets_std[mode], bottom=targets_mean[mode]-targets_std[mode], yerr=[ymin, ymax], capsize=5, alpha=0.7, color=col, width=0.3)
                    plt.errorbar(x_positions+dx, targets_mean[mode], color='k', linestyle='', marker='.')


                plt.xlabel("Targets")
                plt.ylabel("Scores")
                plt.title(f"Score for each traget in {Args.test}")
                plt.xticks(x_positions, targets_labels, rotation=90)
                plt.grid(axis='y', linestyle='--')
                plt.ylim(0)
                plt.tight_layout()
                plt.savefig(f"{Paths.save}/for_target.png")
                plt.close()





        with open(f"{Paths.save}/resume.txt", "w") as f:

            for mode in ["classic", "norma"]:

                f.write(f"{mode}={np.mean(res[mode])}~{np.std(res[mode])}\n")


    print(f"All time : {time()-total_time}")