import os, sys, shutil
import numpy as np
from tqdm import tqdm




def recup_mt(mode="dispo"):


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
                    models.append(f"{mn}_{state_name}")


    else:

        raise Exception(f"Mode {mode} unknow")



    return list(set(models))








def generate_html_table(colonnes, lignes, text, y):


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


def make_score(name_tests, tests, models, score_type, pbar):

    

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

            tot_mean = [list(), list()]
            tot_std = [list(), list()]

            for t, test in enumerate(tests):

                test = otest if "no0" not in model else f"{test}no0"

                pbar.update(1)

                # print(model, test)

                if f"pred_{model}" in os.listdir(f"{path_analyse}/{score}") and test in os.listdir(f"{path_analyse}/{score}/pred_{model}"):

                    # print(f"Analyse {score} > {model} -> {test}")

                    with open(f"{path_analyse}/{score}/pred_{model}/{test}/resume.txt", "r") as f:
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


            for i in range(2):

                mom = np.mean(tot_mean[i])
                soa = np.sum(np.array(tot_std[i])**2)**0.5
                y[i, m, -3] = mom
                e[i, m, -3] = soa
                x[i, m, -3] = f"{mom:.2f} ~ {soa:.2f}"

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



        with open(f"{path_resume}/{name_tests}_{score}.html", "w") as f:

            html_codes = [f"<h1>Score {score}</h1>"]

            for i, typeScore in enumerate(["classic", "norma"]):

                html_codes.append(f"<h2>{typeScore}</h2>")
                html_codes.append(generate_html_table(tests+["Total", "Classement (N)", "Classement (%)"], models, x[i], y[i]))

            f.write('\n'.join(html_codes))



if __name__ == "__main__":

    score_type = ["L1", "chi2"]
    path_analyse = f"./results/analyse"
    path_resume = f"{path_analyse}/all_resume"

    if 'all_resume' in os.listdir(path_analyse) : shutil.rmtree(path_resume)
    os.makedirs(path_resume, exist_ok=True)



    if "local" in sys.argv : tests, nb_ft = {"classic" : ["test64"], "calib" : ["test64calib"]}, 2
    else : tests, nb_ft = {"classic" : ["test1k", "test1kExt", "test1kOT"], "calib" : ["test1kcalib"]}, 4

    mode = "dispo" if "all" not in sys.argv else "all"
    models = recup_mt(mode)

    pbar = tqdm(total=nb_ft*len(models)*2)

    for name, test in tests.items():

        make_score(name, test, models, score_type, pbar)

    pbar.close()




