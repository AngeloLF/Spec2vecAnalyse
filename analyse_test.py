import numpy as np
import matplotlib.pyplot as plt

import os, sys, pickle, json, shutil
import coloralf as c

from time import time




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

    if name == "L1"   : return compute_score_L1(true, pred, give_norm_array=give_norm_array)
    if name == "chi2" : return compute_score_chi2(true, pred, give_norm_array=give_norm_array)


def makeOneSpecOldStyle(fold, select_model, path_train, select_train, num_spec, varp, path4save, give_norma, give_image, savename, score_type):

    with open(f"{fold}/hparams.json", 'r') as f:
        hparams = json.load(f)

    x = np.arange(hparams["LAMBDA_MIN"], hparams["LAMBDA_MAX"], hparams["LAMBDA_STEP"])
    n = int(num_spec)

    pred = np.load(f"{fold}/pred_{select_model}_{select_train}_{select_lr}/spectrum_{num_spec}.npy")
    true = np.load(f"{fold}/spectrum/spectrum_{num_spec}.npy")
    image = np.load(f"{fold}/image/image_{num_spec}.npy")

    flux = np.sum(true) # flux in adu ~

    tn, pn, s, sn = compute_score(score_type, true, pred, give_norm_array=True)
    if give_norma : true, pred = tn, pn

    plt.figure(figsize=(16, 8))
    if give_image : plt.subplot(211)

    plt.plot(x, true, c='g', label='True')
    plt.plot(x, pred, c='r', label='Pred')

    plt.title(f"For {select_model} train with {select_train}_{select_lr} : {s*100:.1f} % [non norma] | {sn*100:.1f} % [norma] | Flux : {flux/1000:.0f} kADU")
    plt.scatter([], [], marker='d', label=f"Target : {varp['TARGET'][n]}", color='k')
    for key, val in varp.items():
        if key != "TARGET" : plt.scatter([], [], marker='*', label=f"{key} = {val[n]:.2f}", color='k')
    plt.legend() 
    plt.xlabel(f"$\lambda$ (nm)")
    plt.ylabel(f"{fold}/*/spectrum_{num_spec}.npy")
    
    if give_image:
        plt.subplot(212)
        plt.imshow(np.log10(image+1), cmap='gray')
        # plt.colorbar()
        plt.savefig(f"{path4save}/example_image/{savename}.png")
        plt.close()
    else:
        plt.savefig(f"{path4save}/example_spectrum/{savename}.png")
        plt.close()


def makeOneSpec(fold, select_model, path_train, select_train, res, varp, n, 
                path4save, give_norma, give_image, savename, score_type):

    with open(f"{fold}/hparams.json", 'r') as f:
        hparams = json.load(f)

    x = np.arange(hparams["LAMBDA_MIN"], hparams["LAMBDA_MAX"], hparams["LAMBDA_STEP"])
    num_spec = res["num"][n]

    pred = np.load(f"{fold}/pred_{select_model}_{select_train}_{select_lr}/spectrum_{num_spec}.npy")
    true = np.load(f"{fold}/spectrum/spectrum_{num_spec}.npy")
    image = np.load(f"{fold}/image/image_{num_spec}.npy")

    tn, pn, s, sn = compute_score(score_type, true, pred, give_norm_array=True)
    if give_norma : true, pred = tn, pn

    plt.figure(figsize=(16, 8))
    if give_image : plt.subplot(211)

    plt.plot(x, true, c='g', label='True')
    plt.plot(x, pred, c='r', label='Pred')

    plt.title(f"For {select_model} train with {select_train}_{select_lr} : {s*100:.1f} % [non norma] | {sn*100:.1f} % [norma] | Flux : {res['flux'][n]/1000:.0f} kADU")
    plt.scatter([], [], marker='d', label=f"Target : {varp['TARGET'][n]}", color='k')
    for key, val in varp.items():
        if key != "TARGET" : plt.scatter([], [], marker='*', label=f"{key} = {val[n]:.2f}", color='k')
    plt.legend() 
    plt.xlabel(f"$\lambda$ (nm)")
    plt.ylabel(f"{fold}/*/spectrum_{num_spec}.npy")
    
    if give_image:
        plt.subplot(212)
        plt.imshow(np.log10(image+1), cmap='gray')
        # plt.colorbar()
        plt.savefig(f"{path4save}/example_image/{savename}.png")
        plt.close()
    else:    
        plt.savefig(f"{path4save}/example_spectrum/{savename}.png")
        plt.close()





def open_fold_classico(fold, pred_folder, path4save, train_params, select_model, select_train, path_train, score_type, cmap="Reds", vmax=0.2, nb_level=10):

    """
    dict res : 
        * keys `classic` & `norma` give a matrice (n_target, n_var)
        * axis 0 : len of n_target
        * axis 1 : len of n_var
    """

    files = os.listdir(f"{fold}/spectrum")
    fold_var = fold.split("test_")[-1]

    with open(f"{fold}/hist_params.json", 'r') as f:
        params = json.load(f)

    with open(f"{fold}/variable_params.pck", "rb") as f:
        var_params = pickle.load(f)

    res = {"classic" : np.zeros(params["nb_simu"]),
           "norma"   : np.zeros(params["nb_simu"]),
           "flux"    : np.zeros(params["nb_simu"]),
           "file"    : np.zeros(params["nb_simu"]).astype(str),
           "num"     : np.zeros(params["nb_simu"]).astype(str)}

    # loading of each spectrum
    for i, file in enumerate(files):

        num_spec_str = file.split("_")[-1][:-4]

        pred = np.load(f"{fold}/{pred_folder}/{file}")
        true = np.load(f"{fold}/spectrum/{file}")

        res["flux"][i] = np.sum(true) # flux in adu ~

        score, score_norma = compute_score(score_type, true, pred)
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
    plt.savefig(f"{path4save}/flux2score.png")
    plt.close()



    for mode in ["classic", "norma"]:
        isNorma = True if mode == "norma" else False

        for i, level in enumerate(np.linspace(np.min(res[mode]), np.max(res[mode]), nb_level)):

            near = np.argmin(np.abs(res[mode]-level))
            makeOneSpec(fold, select_model, path_train, select_train, res, var_params, near, path4save=path4save, give_norma=isNorma, give_image=False, savename=f"Level{i}_{mode}", score_type=select_score_type)
            makeOneSpec(fold, select_model, path_train, select_train, res, var_params, near, path4save=path4save, give_norma=isNorma, give_image=True, savename=f"Level{i}_{mode}", score_type=select_score_type)




    return res, var_params






def open_fold(fold, pred_folder, path4save, train_params, select_model, select_train, path_train, score_type, cmap="Reds", vmax=0.2):

    """
    dict res : 
        * keys `classic` & `norma` give a matrice (n_target, n_var)
        * axis 0 : len of n_target
        * axis 1 : len of n_var
    """

    files = os.listdir(f"{fold}/spectrum")
    fold_var = fold.split("test_")[-1]

    with open(f"{fold}/hist_params.json", 'r') as f:
        params = json.load(f)

    with open(f"{fold}/variable_params.pck", "rb") as f:
        var_params = pickle.load(f)

    n_target = len(params["target_set"])
    n_var = int(params["nb_simu"] / n_target)

    # If 
    if fold_var in params.keys(): 
        var_name = fold_var
        X = np.linspace(*params[var_name], n_var)
    else:
        var_name = f"{fold_var}"
        X = np.linspace(0, 1, n_var)

    res = {"classic" : np.zeros((n_target, n_var)),
           "norma" : np.zeros((n_target, n_var))}

    maxScore = {"classic":[-np.inf, None, None, None], "norma":[-np.inf, None, None, None]}
    minScore = {"classic":[+np.inf, None, None, None], "norma":[+np.inf, None, None, None]}
    # loading of each spectrum
    for file in files:

        num_spec_str = file.split("_")[-1][:-4]
        num_spec = int(num_spec_str)
        num_target = num_spec // n_var
        num_var = num_spec % n_var

        pred = np.load(f"{fold}/{pred_folder}/{file}")
        true = np.load(f"{fold}/spectrum/{file}")

        score, score_norma = compute_score(score_type, true, pred)
        res["classic"][num_target, num_var] = score
        res["norma"][num_target, num_var] = score_norma

        if score > maxScore["classic"][0] : maxScore["classic"] = [score, file, [num_var, num_target], num_spec_str]
        if score < minScore["classic"][0] : minScore["classic"] = [score, file, [num_var, num_target], num_spec_str]
        if score_norma > maxScore["norma"][0] : maxScore["norma"] = [score_norma, file, [num_var, num_target], num_spec_str]
        if score_norma < minScore["norma"][0] : minScore["norma"] = [score_norma, file, [num_var, num_target], num_spec_str]

    # Calcul for y ticks
    delta = X[-1] - X[0]
    dX = delta / n_var
    dY = delta / n_target
    Y = X[0] + dY/2 + np.arange(n_target) * dY

    # Each spectrum
    plt.figure(figsize=(16, 8))

    plt.subplot(221)
    plt.imshow(res["classic"]*100, cmap=cmap, extent=[X[0], X[-1], X[0], X[-1]], origin='lower', vmin=0.0, vmax=vmax*100)
    plt.colorbar()
    plt.yticks(Y, params["target_set"], rotation=0)
    plt.ylabel("Score `classic` (%)")
    showTrainParams(train_params, var_name)
    plotMinMaxScore(X, Y, minScore["classic"][2], maxScore["classic"][2], dX)
    plt.title(f"{minScore['classic'][1]}={minScore['classic'][0]*100:.1f}%  |  {maxScore['classic'][1]}={maxScore['classic'][0]*100:.1f}%")

    plt.subplot(223)
    plt.imshow(res["norma"]*100, cmap=cmap, extent=[X[0], X[-1], X[0], X[-1]], origin='lower', vmin=0.0, vmax=vmax*100)
    plt.colorbar()
    plt.yticks(Y, params["target_set"], rotation=0)
    plt.xlabel(var_name)
    plt.ylabel("Score `norma` (%)")
    showTrainParams(train_params, var_name)
    plotMinMaxScore(X, Y, minScore["norma"][2], maxScore["norma"][2], dX)
    plt.title(f"{minScore['norma'][1]}={minScore['norma'][0]*100:.1f}%  |  {maxScore['norma'][1]}={maxScore['norma'][0]*100:.1f}%")


    # Sum axis target
    plt.subplot(122)
    for mode, col in [("classic", "g"), ("norma", "r")]:
        res_var = np.mean(res[mode], axis=0)*100
        res_var_std = np.std(res[mode], axis=0)*100
        plt.fill_between(X, res_var, res_var+res_var_std, color=col, alpha=0.5)
        plt.fill_between(X, res_var, res_var-res_var_std, color=col, alpha=0.5)
        plt.plot(X, res_var, color=col, label=f"{mode} : {np.mean(res[mode])*100:.1f} $\pm$ {np.std(res[mode])*100:.1f} %")
    showTrainParams(train_params, var_name)
    plt.xlabel(var_name)
    plt.ylabel(f"Score (%)")
    plt.legend()
    plt.title(f"Score for variation of {var_name}")
    plt.ylim(0, vmax*100)
    plt.savefig(f"{path4save}/metric/{var_name}.png")
    plt.close()

    for mode in ["classic", "norma"]:
        isNorma = True if mode == "norma" else False
        makeOneSpecOldStyle(fold, select_model, path_train, select_train, maxScore[mode][3], var_params, path4save=path4save, give_norma=isNorma, give_image=False, savename=f"max_{var_name}_{mode}", score_type=select_score_type)
        makeOneSpecOldStyle(fold, select_model, path_train, select_train, maxScore[mode][3], var_params, path4save=path4save, give_norma=isNorma, give_image=True, savename=f"max_{var_name}_{mode}", score_type=select_score_type)
        makeOneSpecOldStyle(fold, select_model, path_train, select_train, minScore[mode][3], var_params, path4save=path4save, give_norma=isNorma, give_image=False, savename=f"min_{var_name}_{mode}", score_type=select_score_type)
        makeOneSpecOldStyle(fold, select_model, path_train, select_train, minScore[mode][3], var_params, path4save=path4save, give_norma=isNorma, give_image=True, savename=f"min_{var_name}_{mode}", score_type=select_score_type)


    return res, params["target_set"]



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

    total_time = time()

    select_model = None
    select_train = None
    select_test = None
    select_lr = None
    select_score_type = None

    for argv in sys.argv[1:]:

        if argv[:6] == "model=" : select_model = argv[6:]
        if argv[:6] == "train=" : select_train = argv[6:]
        if argv[:5] == "test=" : select_test = argv[5:]
        if argv[:3] == "lr=" : select_lr = f"{float(argv[3:]):.0e}"
        if argv[:6] == "score=" : select_score_type = argv[6:]

    # Define model
    if select_model is None:
        print(f"{c.r}WARNING : model name is not define (model=<select_model>){c.d}")
        raise ValueError("Model name error")

    # Define train folder
    if select_train is None:
        print(f"{c.r}WARNING : train folder is not define (train=<select_train>){c.d}")
        raise ValueError("Train folder error")

    # Define test folder
    if select_test is None:
        print(f"{c.r}WARNING : test folder is not define (test=<select_test>){c.d}")
        raise ValueError("Test folder error")

    # Define test folder
    if select_lr is None:
        print(f"{c.r}WARNING : lr is not define (lr=<lr>){c.d}")
        raise ValueError("Lr error")

    # Define test folder
    if select_score_type is None:
        print(f"{c.r}WARNING : Score type is not define (score=<score_type>){c.d}")
        raise ValueError("Score type error")

    # Define path test
    if "output" in select_test : path_test = "./results"
    else : path_test = "./results/output_simu"

    path_train = './results/output_simu'

    path_save = f"./results/analyse/{select_score_type}"
    pred_folder = f"pred_{select_model}_{select_train}_{select_lr}"

    os.makedirs(f"./results/analyse", exist_ok=True)
    os.makedirs(f"./results/analyse/{select_score_type}", exist_ok=True)
    os.makedirs(f"{path_save}/{pred_folder}", exist_ok=True) 
    

    # chargement des params du train 
    with open(f"{path_train}/{select_train}/hist_params.json", 'r') as f:
        train_params = json.load(f)



    # For each folder test (like output construct with lsp)
    test_folder = select_test

    path4save = f"{path_save}/{pred_folder}/{test_folder}"
    if test_folder in os.listdir(f"{path_save}/{pred_folder}") : shutil.rmtree(f"{path_save}/{pred_folder}/{test_folder}")
    os.mkdir(f"{path_save}/{pred_folder}/{test_folder}")
    os.mkdir(f"{path_save}/{pred_folder}/{test_folder}/metric")
    os.mkdir(f"{path_save}/{pred_folder}/{test_folder}/example_spectrum")
    os.mkdir(f"{path_save}/{pred_folder}/{test_folder}/example_image")


    if test_folder == "output_test":

        sub_folds = [f"{path_test}/{test_folder}/{sf}" for sf in os.listdir(f"{path_test}/{test_folder}") if "test" in sf]
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
        plt.title(f"Score for each traget in {test_folder}")
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
        res, var = open_fold_classico(fold=f"{path_train}/{test_folder}", pred_folder=pred_folder, path4save=path4save, train_params=train_params, 
                select_model=select_model, select_train=select_train, path_train=path_train, score_type=select_score_type)
        print(time()-t0)



        for key, val in var.items():

            if key != "TARGET":

                plt.scatter(val, res["classic"]*100, c="g", label="Classic", alpha=0.6)
                plt.scatter(val, res["norma"]*100, c="r", label="Norma", alpha=0.6)

                plt.legend()
                plt.xlabel(f"Variable {key}")
                plt.ylabel(f"Score (%)")
                plt.savefig(f"{path4save}/metric/{key}.png")
                plt.close()

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
                plt.title(f"Score for each traget in {test_folder}")
                plt.xticks(x_positions, targets_labels, rotation=90)
                plt.grid(axis='y', linestyle='--')
                plt.ylim(0)
                plt.tight_layout()
                plt.savefig(f"{path4save}/for_target.png")
                plt.close()





        with open(f"{path4save}/resume.txt", "w") as f:

            for mode in ["classic", "norma"]:

                f.write(f"{mode}={np.mean(res[mode])}~{np.std(res[mode])}\n")


    print(f"All time : {time()-total_time}")