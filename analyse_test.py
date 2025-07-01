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




def compute_score_L1(true, pred, sim, num_spec_str):

    t = np.copy(true)
    p = np.copy(pred)
    fact_n = np.max(t) / np.max(p)

    p[p < 0] = 0
    t[t < 0] = 0

    # non norma
    diff = np.abs(t-p)
    score = np.sum(diff) / np.sum(t)

    # norma
    pn = p * fact_n
    diff_norma = np.abs(t-pn)
    score_norma = np.sum(diff_norma) / np.sum(t)

    # Save if needed
    return {'score' : score, 
            'score_norma' : score_norma,
            'fact' : fact_n}



def compute_score_chi2(true, pred, sim, num_spec_str, Cread=12, gain=3, SpectractorProcess=False):

    fact_n = np.max(true) / np.max(pred)
    pred_n = pred * fact_n
    sigma_READ = Cread / gain

    true_simu = np.load(f"{Paths.test}/image/image_{num_spec_str}.npy")
    pred_simu,   _, xc, yc = sim.makeSim(num_simu=num_spec_str, updateParams=False, giveSpectrum=pred,   with_noise=False)
    pred_simu_n, _, xc, yc = sim.makeSim(num_simu=num_spec_str, updateParams=False, giveSpectrum=pred_n, with_noise=False)
    arg_timbre = [int(np.round(np.max(f_arg(sim.lambdas, *arg)))) for f_arg, arg in zip(sim.psf_function['f_arg'], sim.psf_function['arg'])]
    timbre_size = sim.psf_function['timbre'](*arg_timbre)

    mask = np.zeros_like(true_simu)
    for xi, yi in zip(xc, yc):
        mask[int(max(0, yi-timbre_size)):int(min(true_simu.shape[0], yi+timbre_size)), int(max(0, xi-timbre_size)):int(min(true_simu.shape[1], xi+timbre_size))] = 1
    true_simu[~(mask == 1)] = 0
    pred_simu[~(mask == 1)] = 0
    pred_simu_n[~(mask == 1)] = 0
    N = np.sum(mask)

    # non-norma
    residus = true_simu - pred_simu
    chi2eq = residus**2 / (sigma_READ**2 + true_simu / gain) * np.sign(residus)
    score = np.sum(np.abs(chi2eq)) / N

    # norma
    residus_n = true_simu - pred_simu_n
    chi2eq_n = residus_n**2 / (sigma_READ**2 + true_simu / gain) * np.sign(residus_n)
    score_n = np.sum(np.abs(chi2eq_n)) / N

    # reduc for spectractor
    if Args.model == "Spectractor" and score > 50 and SpectractorProcess:
        score = np.nan
        score_n = np.nan


    return {'score' : score,
            'score_norma' : score_n,
            'fact' : fact_n,
            'image' : true_simu, 
            'residus' : [residus, residus_n],
            'chi2eq' : [chi2eq, chi2eq_n]}



def compute_score(name, true, pred, sim, num_spec_str):

    if   name == "L1"   : return compute_score_L1(true, pred, sim, num_spec_str)
    elif name == "chi2" : return compute_score_chi2(true, pred, sim, num_spec_str)
    else : raise Exception(f"Unknow score name {name} in analyse_test.compute_score.")

 



def makeOneSpec(true, pred, sim, varp, num_str, give_norma, savename, gain=3.):

    n = int(num_str)

    # compute score
    result = compute_score(Args.score, true, pred, sim, num_str)
    resultChi2 = compute_score("chi2", true, pred, sim, num_str)
    s, sn = result["score"], result["score_norma"]
    if give_norma : pred *= result["fact"]
    true_image = np.load(f"{Paths.test}/image/image_{num_str}.npy")

    # Make subtitle
    pre_title = f"For {Args.model_loss} train with {Args.fulltrain_str}_{Args.lr_str}"
    if Args.score in ["L1"] : title_score = f"{s*100:.1f} % [non norma] | {sn*100:.1f} % [norma]"
    else : title_score = f"{s:.6f} [non norma] | {sn:.6f} [norma]"
    post_title = f"Flux : {np.sum(true) / gain /1000:.0f} kADU" # e- / (e-/ADU) = flux in ADU
    fulltitle = f"{pre_title}, {Args.score} : {title_score} | {post_title}"
         

    # PLOT "RES"
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(16, 8), gridspec_kw={'height_ratios': [3, 1]})
    plt.suptitle(fulltitle)
    plot_spec(ax1, true, pred, varp, num_str)
    plot_res(ax2, true, pred)
    plt.savefig(f"{Paths.save}/figure_res/{savename}.png")
    plt.close()


    # PLOT "res_norma"
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(16, 8), gridspec_kw={'height_ratios': [3, 1]})
    plt.suptitle(fulltitle)
    plot_spec(ax1, true, pred, varp, num_str)
    plot_res(ax2, true, pred, norma=True)
    plt.savefig(f"{Paths.save}/figure_res_norma/{savename}.png")
    plt.close()


    # PLOT "IMAGE"
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=False, figsize=(16, 8), gridspec_kw={'height_ratios': [1, 1]})
    plt.suptitle(fulltitle)
    plot_spec(ax1, true, pred, varp, num_str)
    plot_image(ax2, true_image)
    plt.savefig(f"{Paths.save}/figure_image/{savename}.png")
    plt.close()


    # PLOT "chi2eq"
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(16, 8), gridspec_kw={'height_ratios': [1, 1, 1]})
    plt.suptitle(fulltitle)
    plot_image(ax1, true_image)
    plot_res2d(ax2, resultChi2['residus'][give_norma])
    plot_chi2eq(ax3, resultChi2['chi2eq'][give_norma])

    plt.tight_layout()
    plt.subplots_adjust(hspace=0)
    plt.savefig(f"{Paths.save}/figure_chi2eq/{savename}.png")
    plt.close()


    # PLOT "full"
    # fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(16, 8), gridspec_kw={'height_ratios': [1, 1, 1]})
    # plt.suptitle(fulltitle)
    # plot_image(ax1, true_image)
    # plot_res2d(ax2, resultChi2['residus'][give_norma])
    # plot_chi2eq(ax3, resultChi2['chi2eq'][give_norma])
    # plt.savefig(f"{Paths.save}/figure_full/{savename}.png")
    # plt.close()

    fig = plt.figure(figsize=(20, 8))
    outer = gridspec.GridSpec(1, 2, width_ratios=[1, 1])
    gs1 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[0], height_ratios=[3, 1])
    gs2 = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=outer[1], height_ratios=[1, 1, 1])
    ax1, ax2, ax3, ax4, ax5 = plt.subplot(gs1[0]), plt.subplot(gs1[1]), plt.subplot(gs2[0]), plt.subplot(gs2[1]), plt.subplot(gs2[2])
    plt.suptitle(fulltitle)

    plot_spec(ax1, true, pred, varp, num_str)
    plot_res(ax2, true, pred)

    plot_image(ax3, true_image)
    plot_res2d(ax4, resultChi2['residus'][give_norma])
    plot_chi2eq(ax5, resultChi2['chi2eq'][give_norma])

    plt.subplots_adjust(hspace=0)
    plt.tight_layout()
    plt.savefig(f"{Paths.save}/figure_full/{savename}.png")
    plt.close()



def plot_spec(ax, yt, yp, varp, num_str):

    x = Args.wl
    n = int(num_str)

    ax.plot(x, yt, c='g', label='True')
    ax.plot(x, yp, c='r', label='Pred')
    plot_full_legend(ax, varp, n)
    ax.legend(fontsize=8)
    ax.set_xlabel(f"$\lambda$ (nm)")
    ax.set_ylabel(f"{Paths.test}/*/{Args.folder_output}_{num_str}.npy")

def plot_res(ax, yt, yp, norma=False):

    x = Args.wl
    res = yt - yp
    vmaxs = np.max([yt, yp, np.ones_like(yt)], axis=0)
    
    if norma : res /= vmaxs
    ylabel = "Residus" if not norma else "Residus (norma)"

    ax.axhline(0, color='k', linestyle=':')
    ax.plot(x, res, '.k')
    ax.set_xlabel(f"$\lambda$ (nm)")
    ax.set_ylabel(ylabel)

    if norma :
        ax.set_ylim(-1, 1)

def plot_image(ax, image):

    ax.imshow(np.log10(image+1), cmap='gray')
    ax.set_ylabel(f"Input image")
    ax.set_xticks([]) # Remove x-ticks
    ax.set_yticks([]) # Remove y-ticks

def plot_res2d(ax, residus2d):

    vmax = max(np.abs(np.min(residus2d)), np.max(residus2d))
    ax.imshow(residus2d, cmap='bwr', vmin=-vmax/2, vmax=vmax/2)
    ax.set_ylabel(f"Residus")
    ax.set_xticks([]) # Remove x-ticks
    ax.set_yticks([]) # Remove y-ticks

def plot_chi2eq(ax, chi2eq):

    vmax = max(np.abs(np.min(chi2eq)), np.max(chi2eq))
    title = "\\frac{res}{\\Vert res \\Vert} \\cdot \\frac{res^2}{\\sigma^2_{Read} + it / \\sigma_{gain}}"
    ax.imshow(chi2eq, cmap='bwr', vmin=-vmax/2, vmax=vmax/2)
    ax.set_ylabel(f"${title}$")
    ax.set_xticks([]) # Remove x-ticks
    ax.set_yticks([]) # Remove y-ticks








def open_fold(args, paths, folds, nb_level=5):

    """
    dict res : 
        * keys `classic` & `norma` give a matrice (n_target, n_var)
        * axis 0 : len of n_target
        * axis 1 : len of n_var
    """

    files = os.listdir(f"{paths.test}/{Args.folder_output}")

    with open(f"{paths.test}/hist_params.json", 'r') as f:
        params = json.load(f)

    with open(f"{paths.test}/hparams.json", "r") as f:
        hp = json.load(f)

    with open(f"{paths.test}/variable_params.pck", "rb") as f:
        vp = pickle.load(f)

    res = {"classic" : np.zeros(params["nb_simu"]) * np.nan,
           "norma"   : np.zeros(params["nb_simu"]) * np.nan,
           "flux"    : np.zeros(params["nb_simu"]),
           "file"    : np.zeros(params["nb_simu"]).astype(str),
           "num"     : (np.zeros(params["nb_simu"]) * np.nan).astype(str)}



    # Initialise simulator
    psf_function = {
        'f' : pf.moffat2d_jit,
        'f_arg' : [pf.simpleLinear, pf.simpleLinear],
        'arg' : [[3.0], [3.0]],
        'order0' : {'amplitude':22900.0, 'arg':[3.0, 2.0]},
        'timbre' : pf.moffat2d_timbre,
    }

    sim = SpecSimulator(psf_function=psf_function, savingFolders=False, target_set="setAll")
    


    # loading of each spectrum
    for i, file in enumerate(files):

        num_spec_str = file.split("_")[-1][:-4]

        # set simulator
        n = int(num_spec_str)
        for param in vp.keys():
            if param[:4] != "arg.": 
                sim.__setattr__(param, vp[param][n])
            else:
                sim.psf_function['arg'][0][0] = vp[param][n]

        pred = np.load(f"{paths.test}/{folds.pred_folder}/{file}")
        true = np.load(f"{paths.test}/{Args.folder_output}/{file}")

        res["flux"][i] = np.sum(true) / hp["CCD_GAIN"] # e- / (e-/ADU) = flux in ADU

        result = compute_score(Args.score, true, pred, sim, num_spec_str)

        res["classic"][i] = result['score']
        res["norma"][i] = result['score_norma']
        res["file"][i] = file
        res["num"][i] = num_spec_str



    # FIGURE flux2score
    plt.figure(figsize=(12, 8))
    for mode, col in [("classic", "g"), ("norma", "r")]:
        plt.plot(res["flux"]/1000, res[mode], color=col, label=mode, linestyle="", marker=".", alpha=0.7)
    plt.xlabel("Flux (en kADU)")
    plt.ylabel(f"Score")
    plt.xscale("log")
    plt.legend()
    plt.savefig(f"{Paths.save}/flux2score.png")
    plt.close()



    # FIGURE hist_score
    plt.figure(figsize=(12, 8))
    vmin = min(np.nanmin(res["classic"]), np.nanmin(res["norma"]))
    vmax = max(np.nanmax(res["classic"]), np.nanmax(res["norma"]))
    for mode, col in [("classic", "g"), ("norma", "r")]:
        plt.hist(res[mode], bins=50, range=(vmin, vmax), color=col, alpha=0.8)
    plt.xlabel(f"Score")
    plt.yscale("log")
    plt.legend()
    plt.savefig(f"{Paths.save}/hist_score.png")
    plt.close()    



    

    # 10 exemple of scores
    for mode in ["classic", "norma"]:

        isNorma = True if mode == "norma" else False

        RESMIN = np.min(res[mode][~np.isnan(res[mode])])
        RESMAX = np.max(res[mode][~np.isnan(res[mode])])
        true_levels = np.copy(res[mode])
        true_levels[np.isnan(res[mode])] = np.inf

        for i, level in enumerate(np.linspace(RESMIN, RESMAX, nb_level)):

            near = np.argmin(np.abs(true_levels-level))
            num_spec = res["num"][near]
            n = int(num_spec)

            pred = np.load(f"{Paths.test}/{Folds.pred_folder}/{Args.folder_output}_{num_spec}.npy")
            true = np.load(f"{Paths.test}/{Args.folder_output}/{Args.folder_output}_{num_spec}.npy")

            # set sim
            for param in vp.keys():
                if param[:4] != "arg.": 
                    sim.__setattr__(param, vp[param][n])
                else:
                    sim.psf_function['arg'][0][0] = vp[param][n]

            makeOneSpec(true, pred, sim, vp, num_spec, give_norma=isNorma, savename=f"Level{i}_{mode}")



    return res, vp




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



def plot_full_legend(ax, varp, n):

    ax.scatter([], [], marker='d', label=f"Target : {varp['TARGET'][n]}", color='k')
    for key, val in varp.items():
        if key != "TARGET":
            skey = key if key not in tradargs.keys() else tradargs[key]
            ax.scatter([], [], marker='*', label=f"{skey} = {val[n]:.2f}", color='k')



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
    os.mkdir(f"{Paths.save}/metric")
    os.mkdir(f"{Paths.save}/figure_res")
    os.mkdir(f"{Paths.save}/figure_res_norma")
    os.mkdir(f"{Paths.save}/figure_image")
    os.mkdir(f"{Paths.save}/figure_chi2eq")
    os.mkdir(f"{Paths.save}/figure_full")



    res, var = open_fold(Args, Paths, Folds)



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

            xbin = np.zeros(nbins)
            ybin = np.zeros(nbins)
            ystd = np.zeros(nbins)

            for i in range(nbins):

                resi = yscore[(val > x0 + i*dmax) & (val < x0 + (i+1)*dmax)]

                xbin[i] = x0 + dmax * (0.5 + i)
                ybin[i] = np.mean(resi)
                ystd[i] = np.std(resi)

            plt.figure(figsize=(16, 10))
            plt.scatter(val, yscore, color='k', marker='+', alpha=0.5)
            plt.errorbar(xbin, ybin, yerr=ystd, color="r", linestyle="", marker=".")
            showTrainParams(train_params, var_name=key)
            plt.legend()
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
            # plt.ylim(0)
            plt.tight_layout()
            plt.savefig(f"{Paths.save}/for_target.png")
            plt.close()





    with open(f"{Paths.save}/resume.txt", "w") as f:

        for mode in ["classic", "norma"]:

            true_res = np.copy(res[mode])[~np.isnan(res[mode])]

            f.write(f"{mode}={np.mean(true_res)}~{np.std(true_res)}\n")