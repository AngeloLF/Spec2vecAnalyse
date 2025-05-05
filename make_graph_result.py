import numpy as np
import matplotlib.pyplot as plt

import os, sys, pickle, json, shutil
import coloralf as c




def compute_score(true, pred, give_norm_array=False):

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



def makeOneSpec(fold, select_model, path_train, select_train, num_spec, 
                path4save, give_norma, give_image, savename):

    with open(f"{fold}/hparams.json", 'r') as f:
        hparams = json.load(f)

    x = np.arange(hparams["LAMBDA_MIN"], hparams["LAMBDA_MAX"], hparams["LAMBDA_STEP"])

    pred = np.load(f"{fold}/pred_{select_model}_{select_train}/spectrum_{num_spec}.npy")
    true = np.load(f"{fold}/spectrum/spectrum_{num_spec}.npy")
    image = np.load(f"{fold}/image/image_{num_spec}.npy")

    tn, pn, s, sn = compute_score(true, pred, give_norm_array=True)
    if give_norma : true, pred = tn, pn

    plt.figure(figsize=(16, 8))
    if give_image : plt.subplot(211)

    plt.plot(x, true, c='g', label='True')
    plt.plot(x, pred, c='r', label='Pred')

    plt.title(f"For {select_model} train with {select_train} : {s*100:.1f} % [non norma] | {sn*100:.1f} % [norma]")
    plt.legend() 
    plt.xlabel(f"$\lambda$ (nm)")
    plt.ylabel(f"{fold}/*/spectrum_{num_spec}.npy")
    
    if give_image:
        plt.subplot(212)
        plt.imshow(np.log10(image+1), cmap='gray')
        # plt.colorbar()
        plt.savefig(f"{path4save}/example_spectrum/{savename}.png")
        plt.close()
    else:
        plt.savefig(f"{path4save}/example_image/{savename}.png")
        plt.close()





def open_fold_classico(fold, pred_folder, path4save, train_params, select_model, select_train, path_train, cmap="Reds", vmax=0.2):

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
           "norma" : np.zeros(params["nb_simu"])}

    maxScore = {"classic":[0.0, None], "norma":[0.0, None]}
    minScore = {"classic":[1.0, None], "norma":[1.0, None]}
    # loading of each spectrum
    for i, file in enumerate(files):

        num_spec_str = file.split("_")[-1][:-4]

        pred = np.load(f"{fold}/{pred_folder}/{file}")
        true = np.load(f"{fold}/spectrum/{file}")

        score, score_norma = compute_score(true, pred)
        res["classic"][i] = score
        res["norma"][i] = score_norma

        if score > maxScore["classic"][0] : maxScore["classic"] = [score, file, None, num_spec_str]
        if score < minScore["classic"][0] : minScore["classic"] = [score, file, None, num_spec_str]
        if score_norma > maxScore["norma"][0] : maxScore["norma"] = [score_norma, file, None, num_spec_str]
        if score_norma < minScore["norma"][0] : minScore["norma"] = [score_norma, file, None, num_spec_str]


    for mode in ["classic", "norma"]:
        isNorma = True if mode == "norma" else False
        makeOneSpec(fold, select_model, path_train, select_train, maxScore[mode][3], path4save=path4save, give_norma=isNorma, give_image=False, savename=f"max_{mode}")
        makeOneSpec(fold, select_model, path_train, select_train, maxScore[mode][3], path4save=path4save, give_norma=isNorma, give_image=True, savename=f"max_{mode}")
        makeOneSpec(fold, select_model, path_train, select_train, minScore[mode][3], path4save=path4save, give_norma=isNorma, give_image=False, savename=f"min_{mode}")
        makeOneSpec(fold, select_model, path_train, select_train, minScore[mode][3], path4save=path4save, give_norma=isNorma, give_image=True, savename=f"min_{mode}")



    return res, var_params






def open_fold(fold, pred_folder, path4save, train_params, select_model, select_train, path_train, cmap="Reds", vmax=0.2):

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

    maxScore = {"classic":[0.0, None], "norma":[0.0, None]}
    minScore = {"classic":[1.0, None], "norma":[1.0, None]}
    # loading of each spectrum
    for file in files:

        num_spec_str = file.split("_")[-1][:-4]
        num_spec = int(num_spec_str)
        num_target = num_spec // n_var
        num_var = num_spec % n_var

        pred = np.load(f"{fold}/{pred_folder}/{file}")
        true = np.load(f"{fold}/spectrum/{file}")

        score, score_norma = compute_score(true, pred)
        res["classic"][num_target, num_var] = score
        res["norma"][num_target, num_var] = score_norma

        if score > maxScore["classic"][0] : maxScore["classic"] = [score, file, [num_var, num_target], num_spec_str]
        if score < minScore["classic"][0] : minScore["classic"] = [score, file, [num_var, num_target], num_spec_str]
        if score_norma > maxScore["norma"][0] : maxScore["norma"] = [score_norma, file, [num_var, num_target], num_spec_str]
        if score_norma < minScore["norma"][0] : minScore["norma"] = [score_norma, file, [num_var, num_target], num_spec_str]

    # Calcul for y ticks
    delta = X[-1] - X[0]
    dX = delta / n_target
    Y = X[0] + dX/2 + np.arange(n_target) * dX

    # Each spectrum
    plt.figure(figsize=(16, 8))

    plt.subplot(221)
    plt.imshow(res["classic"], cmap=cmap, extent=[X[0], X[-1], X[0], X[-1]], origin='lower', vmin=0.0, vmax=vmax)
    plt.colorbar()
    plt.yticks(Y, params["target_set"], rotation=0)
    plt.ylabel("Score `classic`")
    showTrainParams(train_params, var_name)
    plotMinMaxScore(X, Y, minScore["classic"][2], maxScore["classic"][2], dX)
    plt.title(f"{minScore['classic'][1]}={minScore['classic'][0]*100:.1f}%  |  {maxScore['classic'][1]}:{maxScore['classic'][0]*100:.1f}%")

    plt.subplot(223)
    plt.imshow(res["norma"], cmap=cmap, extent=[X[0], X[-1], X[0], X[-1]], origin='lower', vmin=0.0, vmax=vmax)
    plt.colorbar()
    plt.yticks(Y, params["target_set"], rotation=0)
    plt.xlabel(var_name)
    plt.ylabel("Score `norma`")
    showTrainParams(train_params, var_name)
    plotMinMaxScore(X, Y, minScore["norma"][2], maxScore["norma"][2], dX)
    plt.title(f"{minScore['norma'][1]}={minScore['norma'][0]*100:.1f}%  |  {maxScore['norma'][1]}:{maxScore['norma'][0]*100:.1f}%")


    # Sum axis target
    plt.subplot(122)
    for mode, col in [("classic", "g"), ("norma", "r")]:
        res_var = np.mean(res[mode], axis=0)
        res_var_std = np.std(res[mode], axis=0)
        plt.fill_between(X, res_var, res_var+res_var_std, color=col, alpha=0.5)
        plt.fill_between(X, res_var, res_var-res_var_std, color=col, alpha=0.5)
        plt.plot(X, res_var, color=col, label=f"{mode} : {np.mean(res[mode])*100:.1f} $\pm$ {np.std(res[mode])*100:.1f} %")
    showTrainParams(train_params, var_name)
    plt.xlabel(var_name)
    plt.ylabel(f"Score")
    plt.legend()
    plt.title(f"Score for variation of {var_name}")
    plt.ylim(0, 1)
    plt.savefig(f"{path4save}/metric/{var_name}.png")
    plt.close()





    for mode in ["classic", "norma"]:
        isNorma = True if mode == "norma" else False
        makeOneSpec(fold, select_model, path_train, select_train, maxScore[mode][3], path4save=path4save, give_norma=isNorma, give_image=False, savename=f"max_{var_name}_{mode}")
        makeOneSpec(fold, select_model, path_train, select_train, maxScore[mode][3], path4save=path4save, give_norma=isNorma, give_image=True, savename=f"max_{var_name}_{mode}")
        makeOneSpec(fold, select_model, path_train, select_train, minScore[mode][3], path4save=path4save, give_norma=isNorma, give_image=False, savename=f"min_{var_name}_{mode}")
        makeOneSpec(fold, select_model, path_train, select_train, minScore[mode][3], path4save=path4save, give_norma=isNorma, give_image=True, savename=f"min_{var_name}_{mode}")


    return res, params["target_set"]



def showTrainParams(train_params, var_name):
    """
    Pour mettre des axvline pour renseigner de la plage d'entrainement du modèle
    """

    if var_name in train_params.keys():
        minTrain, maxTrain = train_params[var_name]
        plt.axvline(minTrain, color='k', linestyle=':', label='In training')
        plt.axvline(maxTrain, color='k', linestyle=':')

def plotMinMaxScore(X, Y, minXY, maxXY, dX=0):

    plt.scatter(X[minXY[0]]-dX/2, Y[minXY[1]], marker="d", facecolor='white', color='k')
    plt.scatter(X[maxXY[0]]-dX/2, Y[maxXY[1]], marker="d", facecolor='darkred', color='k')



if __name__ == "__main__":

    select_model = None
    select_train = None

    for argv in sys.argv[1:]:

        if argv[:6] == "model=" : select_model = argv[6:].lower()
        if argv[:6] == "train=" : select_train = argv[6:].lower()

    # Define model
    if select_model is None:
        print(f"{c.r}WARNING : model name is not define (model=<select_model>){c.d}")
        raise ValueError("Model name error")

    # Define train folder
    if select_train is None:
        print(f"{c.r}WARNING : train folder is not define (train=<select_train>){c.d}")
        raise ValueError("Train folder error")

    path_test = './results'
    folders_test = ["output_test", "test128"]

    path_train = './results/output_simu'
    name_of_spectrum_folders = "spectrum"

    path_save = "./results/analyse"
    pred_folder = f"pred_{select_model}_{select_train}"

    if pred_folder in os.listdir(path_save) : shutil.rmtree(f"{path_save}/{pred_folder}")
    os.mkdir(f"{path_save}/{pred_folder}")

    # chargement des params du train 
    with open(f"{path_train}/{select_train}/hist_params.json", 'r') as f:
        train_params = json.load(f)



    # For each folder test (like output construct with lsp)

    for i, test_folder in enumerate(folders_test):



        print(f"\n{c.g}Folder Test : {test_folder} [{i+1}/{len(folders_test)}]{c.d}")

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
                    select_model=select_model, select_train=select_train, path_train=path_train)

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


            res, var = open_fold_classico(fold=f"{path_train}/{test_folder}", pred_folder=pred_folder, path4save=path4save, train_params=train_params, 
                    select_model=select_model, select_train=select_train, path_train=path_train)




            for key, val in var.items():

                if key != "TARGET":

                    plt.scatter(res["classic"], val, c="g", label="Classic")
                    plt.scatter(res["norma"], val, c="r", label="Norma")

                    plt.xlabel(f"Variable {key}")
                    plt.ylabel(f"Score (%)")
                    plt.savefig(f"{path4save}/metric/{key}.png")
                    plt.close()

                else:

                    pass


            with open(f"{path4save}/resume.txt", "w") as f:

                for mode in ["classic", "norma"]:

                    f.write(f"For the `{mode}` calcul : {np.mean(res[mode])*100:.2f} ~ {np.std(res[mode])*100:.2f} %\n")

