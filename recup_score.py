import os, sys, shutil
import numpy as np
from tqdm import tqdm
import coloralf as c
import matplotlib.pyplot as plt



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





def addAnalyse(ana, name, listOfCarac):

    ana[]








def generate_html_table(colonnes, lignes, text, y, sorting=False, marker='.', savefig_name=None, markers=None, colors=None):


    if sorting:

        index = np.argsort(y[:, -3])

        y = y[index]
        text = text[index]
        lignes = [lignes[i] for i in index]

        # y[y == np.inf] = np.nan

        for color_palette, palette in colors.items():

            plt.figure(figsize=(19, 10))

            for i, name in enumerate(lignes):

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

            plt.xticks(np.arange(len(lignes)), lignes, rotation=90)
            plt.tight_layout()
            plt.yscale("log")
            plt.legend()
            # plt.show()
            plt.savefig(f"{savefig_name}_{color_palette}.png")
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


def make_score(name_tests, tests, models, score_type, pbar, markers, colors):

    

    for score in score_type:

        if score not in os.listdir(f"{path_analyse}"):
            break

        # Sorting lists
        models.sort()


        y = np.zeros((2, len(models), len(tests)+3)) + np.inf
        e = np.zeros((2, len(models), len(tests)+3)) + np.inf
        x = np.zeros((2, len(models), len(tests)+3)).astype(str)
        x[:, :] = '---'

        
        for m, model in enumerate(models):

            # print(f"{c.ly}Model {model}{c.d}")

            tot_mean = [list(), list()]
            tot_std = [list(), list()]

            for t, otest in enumerate(tests):

                test = otest if "no0" not in model else f"{otest}no0"

                # print(f"{c.y}Test {otest}{c.d}")

                pbar.update(1)

                if f"{model}" in os.listdir(f"{path_analyse}/{score}") and test in os.listdir(f"{path_analyse}/{score}/{model}"):

                    with open(f"{path_analyse}/{score}/{model}/{test}/resume.txt", "r") as f:
                        data = f.read().split("\n")[:-1]

                    for i, line in enumerate(data):

                        try:
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

                            tot_mean[i].append(mean)
                            tot_std[i].append(std)

                        except Exception as err:
                                
                            print(f"\nException : {err} ...")
                            print(f"Error on {data} on {model} -> {test} ...")

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
                    html_codes.append(generate_html_table(tests+["Total", "Classement (N)", "Classement (%)"], models, x[i], y[i], sorting=sorting, savefig_name=f"{path_resume}/graph/{name_tests}", markers=markers, colors=colors))

                f.write('\n'.join(html_codes))



if __name__ == "__main__":

    score_type = ["L1", "chi2"]
    path_analyse = f"./results/analyse"
    path_resume = f"{path_analyse}/all_resume"

    if 'all_resume' in os.listdir(path_analyse) : shutil.rmtree(path_resume)
    os.makedirs(path_resume, exist_ok=True)
    os.makedirs(f"{path_resume}/graph", exist_ok=True)
    os.makedirs(f"{path_resume}/html", exist_ok=True)



    # Special graph
    ANALYSE = {
        "SCaM_by_loss"  : [["SCaM_", "chi2_"], ["SCaM_", "MSE_"], ["SCaM_", "L1N_"]],
        "SCaM_by_train" : [["SCaM_", "train2k_"], ["SCaM_", "train4k_"], ["SCaM_", "train8k_"], ["SCaM_", "train16k_"]],
        "SCaM_by_lr"    : [["SCaM_", "1e-03"], ["SCaM_", "1e-04"], ["SCaM_", "5e-05"], ["SCaM_", "1e-05"], ["SCaM_", "5e-06"], ["SCaM_", "1e-06"]],
    }

    if "local" in sys.argv:
        tests, nb_ft = {"classic" : ["test4", "test5", "test6"]}, 3
        markers = {"classic" : None}
    else: 
        tests, nb_ft = {"classic" : ["test1k", "test1kExt", "test1kOT"]}, 3
        markers = {"classic" : None}

    colors = {
        "model"        : {"SCaM_" : "r", "SCaMv2_":"darkred", "SotSu_" : "b", "SotSuv2_":"darkblue", "CaTS":"g"},
        "metric"       : {"chi2" : "r", "MSE" : "b", "L1N" : "g"},
        "trainingType" : {"kwc_":"darkred", "kwcno0":"r", "kwcPXno0":"b", "kno0":"g", "k_":"gray"},
        "trainNk"      : {"16k" : "g", "8k" : "b", "4k" : "r", "2k" : "gray"},
        "learningRate" : {"1e-06":"b", "1e-05":"g", "5e-05":"yellow", "1e-04":"orange", "1e-03":"red", "1e-02":"darkred"}
    }

    mode = "dispo" if "all" not in sys.argv else "all"
    
    models = list()
    for score in score_type:
        models += recup_mt(score, mode)
    models = list(set(models))

    pbar = tqdm(total=nb_ft*len(models)*len(score_type))

    for name, test in tests.items():

        make_score(name, test, models, score_type, pbar, markers[name], colors)

    pbar.close()




