import os, sys, shutil
import numpy as np
from tqdm import tqdm
import coloralf as c
import matplotlib.pyplot as plt
from types import SimpleNamespace





def recup_mt(score, mode="dispo"):


    if mode == "dispo":

        models = os.listdir(f"./results/analyse/{score}")



    elif mode == "all":

        model_name = os.listdir(f"./results/Spec2vecModels_Results")
        models = list()

        for mn in model_name:
            states = os.listdir(f"./results/Spec2vecModels_Results/{mn}/states")

            for state in states:

                if "_best" in state:

                    state_name = state.split("_best")[0] 

                    if not ("cal" in state and state.count("train") == 1):
                        
                        models.append(f"{mn}_{state_name}")

                    else:

                        print(f"Exclude pre-trained {c.r}{mn}_{state_name}{c.d}")


    else:

        raise Exception(f"Mode {mode} unknow")



    return list(set(models))



def initAnalyse(tests, colors):

    """
    k : key
    n : names
    a : args

    For example:
        SCaM_by_loss = ["chi2", "MSE" "L1N"]

        k2a : SCaM_by_loss -> {"chi2": {"test1k":[float1, float2 ...], "test1kOT":[float1, float2 ...]}, 
                               "MSE" : {"test1k":[float1, float2 ...], "test1kOT":[float1, float2 ...]},
                               "L1N" : {"test1k":[float1, float2 ...], "test1kOT":[float1, float2 ...]}}
        k2p : SCaM_by_loss -> ["SCaM_chi2_train2k_1e-6", ...]
    """

    ANA = SimpleNamespace()
    ANA.k2a = dict() # float mean
    ANA.k2s = dict() # float std
    ANA.k2l = dict() # 
    ANA.k2p = dict() # 
    ANA.tests = tests
    ANA.colors = colors

    return ANA


def addAnalyse(ana, name, inSetPreds, listOfArgs):

    ana.k2a[name] = dict()
    ana.k2s[name] = dict()
    ana.k2l[name] = dict()
    ana.k2p[name] = inSetPreds

    for a in listOfArgs:

        ana.k2a[name][a] = dict()
        ana.k2s[name][a] = dict()
        ana.k2l[name][a] = dict()

        for test in ana.tests : 
            ana.k2a[name][a][test] = list()
            ana.k2l[name][a][test] = list()
            ana.k2s[name][a][test] = list()


def addValueInAnalyse(ana, model, otest, m, s):

    for k, p in ana.k2p.items():

        if model in p:

            for a in ana.k2a[k].keys():

                if a[0] == "~":

                    if a[1:] not in model:

                        ana.k2a[k][a][otest].append(m)
                        ana.k2s[k][a][otest].append(s)
                        ana.k2l[k][a][otest].append(model)

                else:

                    if a in model:

                        ana.k2a[k][a][otest].append(m)
                        ana.k2s[k][a][otest].append(s)
                        ana.k2l[k][a][otest].append(model)



def makePlotAnalyse(ana, score, idec=0.1):

    for k, a in ana.k2a.items():

        ns = list(a.keys())
        x = np.arange(len(ns))

        mean_score_models = np.zeros(len(ns))
        stds_score_models = np.zeros(len(ns))
        mean_score_names  = np.zeros(len(ns)).astype(str)


        score_models = list()
        score_models_std = list()
        for _ in range(len(ns)): 
            score_models.append(dict())
            score_models_std.append(dict())

        plt.figure(figsize=(16, 8))
        decalages = np.linspace(-idec, idec, len(ana.tests) + 1)

        for decalage, test, col in zip(decalages[:-1], ana.tests, ana.colors):
            
            l_min = np.zeros(len(ns)).astype(str)
            y_min = np.zeros(len(ns))
            y_std = np.zeros(len(ns))
            y_mean = np.zeros(len(ns))


            for i, n in enumerate(ns):

                for j, m in enumerate(ana.k2l[k][n][test]):

                    if m not in score_models[i] : score_models[i][m] = list()
                    score_models[i][m].append(a[n][test][j])

                    if m not in score_models_std[i] : score_models_std[i][m] = list()
                    score_models_std[i][m].append(ana.k2s[k][n][test][j])

                l_min[i] = ana.k2l[k][n][test][np.argmin(a[n][test])]
                y_min[i] = np.min(a[n][test])
                y_std[i] = ana.k2s[k][n][test][np.argmin(a[n][test])]
                y_mean[i] = np.mean(a[n][test])

            # plt.plot(x, y_mean, color=col, marker='.', linestyle="", label=f"Mean of {test}")
            plt.errorbar(x+decalage, y_min, yerr=y_std, color=col, marker='*', linestyle="", label=f"Best for {test}")
            plt.axhline(np.min(y_min), color=col, linestyle=":", label=l_min[np.argmin(y_min)])


        for i in range(len(ns)):
            means_score = [np.mean(s) for s in score_models[i].values()]
            stds_score = [np.sum(np.array(s)**2)**0.5/3 for s in score_models_std[i].values()]
            mean_score_models[i] = np.min(means_score)
            stds_score_models[i] = stds_score[np.argmin(means_score)]

            # print(c.r, score_models[i], c.d)
            # print(list(score_models[i].keys()))
            # print(c.g, means_score, c.d)
            # print(c.b, np.argmin(means_score), c.d)

            mean_score_names[i] = list(score_models[i].keys())[np.argmin(means_score)]

        plt.errorbar(x+decalages[-1], mean_score_models, yerr=stds_score_models, color="k", marker='*', linestyle="", label=f"Best average")
        plt.axhline(np.min(mean_score_models), color="k", linestyle=":", label=mean_score_names[np.argmin(mean_score_models)])

        plt.legend()
        plt.title(f"{k}")
        plt.xticks(np.arange(len(ns)), ns)
        plt.yscale("log")
        plt.savefig(f"./results/analyse/all_resume/graph/classic_{score}_{k}.png")
        plt.close()




def generate_html_table(colonnes, lignes, text, y, sorting=False, marker='.', savefig_name=None, markers=None, colors=None, score=None):


    if sorting:

        index = np.argsort(y[:, -3])

        y = y[index]
        text = text[index]
        lignes = [lignes[i] for i in index]

        # y[y == np.inf] = np.nan

        for color_palette, palette in colors.items():

            for zoom, zoom_str in [(lignes, ""), (lignes[:16], "zoom_")]:

                plt.figure(figsize=(19, 10))

                for i, name in enumerate(zoom):

                    xg = np.ones(len(colonnes)-2) * i
                    yg = y[i][:-2]

                    color = 'k'
                    for pal, col in palette.items():
                        if   color_palette != "learningRate" and pal in name      : color = col
                        elif color_palette == "learningRate" and pal == name[-5:] : color = col

                    plt.plot(xg, yg, color=color)

                    plt.scatter([i], yg[-1], color=color, marker="s")
                

                for pal, col in palette.items():
                    plt.scatter([], [], color=col, marker='s', label=pal)

                plt.xticks(np.arange(len(zoom)), zoom, rotation=90)
                plt.tight_layout()
                plt.yscale("log")
                plt.legend()
                # plt.show()
                plt.savefig(f"{savefig_name}_{zoom_str}{score}_{color_palette}.png")
                plt.close()



    tds = {
        "def" : "td",
            
        "far_min" : 'td style="background-color: #CCFFCC;"',
        "near_min" : 'td style="background-color: #66FF66;"',
        "min" : 'td style="background-color: #00CC00; font-weight: bold;"',

        "far_max" : 'td style="background-color: #FFCCCC;"',
        "near_max" : 'td style="background-color: #FF6666;"',
        "max" : 'td style="background-color: #CC0000; font-weight: bold;"',

        "nan" : 'td style="background-color: #000000;"',
    }

    if text.shape != (len(lignes), len(colonnes)):
        raise ValueError("Les dimensions de y ne correspondent pas aux longueurs des listes ligne et colonne.")

    html = '<table border="1" style="border-collapse: collapse; text-align: center;">\n'
    
    # En-tête
    html += '  <tr><th></th>'  # Coin supérieur gauche vide
    for col in colonnes:
        html += f'<th>{col}</th>'
    html += '</tr>\n'


    buffer_y = np.copy(y)
    buffer_y[y == np.inf] = np.nan

    argmin, argmax = np.zeros(buffer_y.shape[1]) + np.nan, np.zeros(buffer_y.shape[1]) + np.nan
    valmin, valmax = np.zeros(buffer_y.shape[1]) + np.nan, np.zeros(buffer_y.shape[1]) + np.nan

    for k in range(buffer_y.shape[1]):

        # print(buffer_y[:, k])

        if not np.all(np.isnan(buffer_y[:, k])):

            argmin[k], argmax[k] = np.nanargmin(buffer_y[:, k]), np.nanargmax(buffer_y[:, k])
            valmin[k], valmax[k] = np.nanmin(buffer_y[:, k]),    np.nanmax(buffer_y[:, k])

    # print(len(argmin), buffer_y.shape)

    # Lignes de données
    for i, ligne in enumerate(lignes):
        html += f'  <tr><th>{ligne}</th>'
        for j in range(len(colonnes)):

            if   i == argmin[j] : td = tds["min"]
            elif i == argmax[j] : td = tds["max"]
            elif y[i, j] < valmin[j] * 1.2 : td = tds["near_min"]
            elif y[i, j] < valmin[j] * 1.5 : td = tds["far_min"]
            elif y[i, j] < valmin[j] / 1.5 : td = tds["far_max"] 
            elif y[i, j] > valmax[j] / 1.2 : td = tds["near_max"] 
            else : td = tds["def"]

            if np.isnan(buffer_y[i, j]) : td = tds["nan"]

            html += f'<{td}>{text[i, j]}</td>'
        html += '</tr>\n'
    
    html += '</table>'
    return html


def make_score(name_tests, tests, models, score_type, pbar, markers, colors, tests_colors):

    

    for score in score_type:

        ana = initAnalyse(tests, tests_colors)

        set1 = possibility(models=["SCaM"], losses=["chi2", "L1N", "MSE"], trains=["train2k", "train4k", "train8k", "train16k"], lrs=["1e-03", "1e-04", "5e-05", "1e-05", "5e-06", "1e-06"])
        addAnalyse(ana, "ANALYSE_SCaM_by_loss",  set1, ["MSE", "L1N", "chi2"])
        addAnalyse(ana, "ANALYSE_SCaM_by_train", set1, ["train2k", "train4k", "train8k", "train16k"])
        addAnalyse(ana, "ANALYSE_SCaM_by_lr",    set1, ["1e-03", "1e-04", "5e-05", "1e-05", "5e-06", "1e-06"])

        set2 = possibility(models=["SCaM"], losses=["chi2"], trains=["train16k", "train16kno0", "train16kwc", "train16kwcno0", "train16kwcPX", "train16kwcPXno0"], lrs=["1e-04", "5e-05", "1e-05", "5e-06"])
        addAnalyse(ana, "ANALYSE_SCaM_by_16k_all",   set2, ["16k_", "16kno0_", "16kwc_", "16kwcno0_", "16kwcPX_", "16kwcPXno0_"])
        addAnalyse(ana, "ANALYSE_SCaM_by_16k_calib", set2, ["~wc", "wc"])
        addAnalyse(ana, "ANALYSE_SCaM_by_16k_no0",   set2, ["~no0", "no0"])

        set3 = possibility(models=["SCaM", "SCaMv2", "SotSu", "SotSuv2", "CaTS", "CaTSv2"], losses=["chi2"], trains=["train16k"], lrs=["1e-04", "5e-05", "1e-05", "5e-06", "1e-6"])
        addAnalyse(ana, "ANALYSE_by_Models", set3, ["SCaM_", "SCaMv2_", "SotSu_", "SotSuv2_", "CaTS_", "CaTSv2_"])


        if score not in os.listdir(f"{path_analyse}"):
            break

        # Sorting lists
        models.sort()


        y = np.zeros((2, len(models), len(tests)+3)) + np.inf
        e = np.zeros((2, len(models), len(tests)+3)) + np.inf
        x = np.zeros((2, len(models), len(tests)+3)).astype(str)
        x[:, :] = '---'

        
        for m, model in enumerate(models):

            tot_mean = [list(), list()]
            tot_std = [list(), list()]

            for t, otest in enumerate(tests):

                test = otest if "no0" not in model else f"{otest}no0"

                pbar.update(1)

                if f"{model}" in os.listdir(f"{path_analyse}/{score}") and test in os.listdir(f"{path_analyse}/{score}/{model}"):

                    with open(f"{path_analyse}/{score}/{model}/{test}/resume.txt", "r") as f:
                        data = f.read().split("\n")[:-1]

                    for i, line in enumerate(data):

                        label, score_i = line.split("=")
                        mean, std = score_i.split("~")
                        mean = float(mean)
                        std = float(std)

                        if score in ["L1"]:
                            mean *= 100
                            std *= 100

                        y[i, m, t] = mean
                        e[i, m, t] = std
                        x[i, m, t] = f"{mean:.2f} ~ {std:.2f}"
                        if score == "L1"     : x[i, m, t] = f"{mean:.2f} ~ {std:.2f}"
                        elif score == "chi2" : x[i, m, t] = f"{mean:.4f} ~ {std:.4f}"
                        else : raise Exception(f"Score {score} unknow")

                        if i == 0 : addValueInAnalyse(ana, model, otest, mean, std)

                        tot_mean[i].append(mean)
                        tot_std[i].append(std)

                else:

                    # print(f"{c.r}Analyse {score} > {model} -> {test} unknow{c.d}")
                    pass


            for i in range(2):

                mom = np.mean(tot_mean[i])
                soa = np.sum(np.array(tot_std[i])**2)**0.5
                y[i, m, -3] = mom
                e[i, m, -3] = soa
                if score == "L1"     : x[i, m, -3] = f"{mom:.2f} ~ {soa:.2f}"
                elif score == "chi2" : x[i, m, -3] = f"{mom:.6f} ~ {soa:.6f}"
                else : raise Exception(f"Score {score} unknow")

        for i in range(2):

            y[i, :, -3][np.isnan(y[i, :, -3])] = np.inf
            nb_m = len(y[i, :, -3])
            order = np.zeros(nb_m)

            for m, cl in enumerate(np.argsort(y[i, :, -3])):

                order[cl] = m

            order_norma = order / (nb_m-1) * 100

            y[i, :, -1] = order_norma + 100
            x[i, :, -1] = [f"{o:.2f} %" for o in order_norma]

            y[i, :, -2] = order_norma + 100
            x[i, :, -2] = [f"{1+o}" for o in order]


        for sorting, sorting_str in [(False, ""), (True, "_sorting")]:

            with open(f"{path_resume}/html/{name_tests}_{score}{sorting_str}.html", "w") as f:

                html_codes = [f"<h1>Score {score}</h1>"]

                for i, typeScore in enumerate(["classic"]):

                    html_codes.append(f"<h2>{typeScore}</h2>")
                    # if score == "L1" and typeScore == "classic":
                    html_codes.append(generate_html_table(tests+["Total", "Classement (N)", "Classement (%)"], models, x[i], y[i], sorting=sorting, savefig_name=f"{path_resume}/graph/{name_tests}", markers=markers, colors=colors, score=score))

                f.write('\n'.join(html_codes))

        makePlotAnalyse(ana, score)




def possibility(models, losses, trains, lrs, loads=[None]):

    preds = list()

    for model in models:
        for loss in losses:
            for train in trains:
                for lr in lrs:
                    for load in loads:

                        pred = f"pred_{model}_{loss}_{train}_{lr}"
                        if load is not None : pred += f"_{load}"
                        preds.append(pred)

    return preds




if __name__ == "__main__":

    score_type = ["L1", "chi2"]
    path_analyse = f"./results/analyse"
    path_resume = f"{path_analyse}/all_resume"

    if 'all_resume' in os.listdir(path_analyse) : shutil.rmtree(path_resume)
    os.makedirs(path_resume, exist_ok=True)
    os.makedirs(f"{path_resume}/graph", exist_ok=True)
    os.makedirs(f"{path_resume}/html", exist_ok=True)

    if "local" in sys.argv:
        tests, nb_ft = {"classic" : ["test4", "test5", "test6"]}, 3
        markers = {"classic" : None}
        tests_colors = {"classic" : ["r", "g", "b"]}
    else: 
        tests, nb_ft = {"classic" : ["test1k", "test1kExt", "test1kOT"]}, 3
        markers = {"classic" : None}
        tests_colors = {"classic" : ["r", "g", "b"]}

    colors = {
        "model"        : {"SCaM_" : "r", "SCaMv2_":"darkred", "SotSu_" : "b", "SotSuv2_":"darkblue", "CaTS":"g"},
        "metric"       : {"chi2" : "r", "MSE" : "b", "L1N" : "g"},
        "trainingType" : {"kwc_":"darkred", "kwcno0":"r", "kwcPXno0":"b", "kno0":"g", "k_":"gray"},
        "trainNk"      : {"16k" : "g", "8k" : "b", "4k" : "r", "2k" : "gray"},
        "learningRate" : {"1e-06":"magenta", "5e-06":"b", "1e-05":"g", "5e-05":"yellow", "1e-04":"orange", "1e-03":"red", "1e-02":"darkred"}
    }

    mode = "dispo" if "all" not in sys.argv else "all"
    
    models = list()
    for score in score_type:
        models += recup_mt(score, mode)
    models = list(set(models))

    pbar = tqdm(total=nb_ft*len(models)*len(score_type))

    for name, test in tests.items():

        make_score(name, test, models, score_type, pbar, markers[name], colors, tests_colors[name])

    pbar.close()




