import numpy as np
import os, sys, json, shutil
import matplotlib.pyplot as plt
from tqdm import tqdm

def generate_html_table(colonnes, lignes, text, y, sorting=False):


    if sorting:

        index = np.argsort(y[:, -2])

        y = y[index]
        text = text[index]
        lignes = [lignes[i] for i in index]


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





if __name__ == "__main__":

    path_analyse = f"./results/analyse/sfopa"
    path_resume = f"./results/analyse/all_resume_FOPA"
    score = "sfopa"
    tests = ["Ozone", "PWV", "Aerosols"]

    if 'all_resume_FOPA' in os.listdir("./results/analyse") : shutil.rmtree(path_resume)
    os.makedirs(path_resume, exist_ok=True)
    os.makedirs(f"{path_resume}/graph", exist_ok=True)
    os.makedirs(f"{path_resume}/html", exist_ok=True)

    models = [m for m in os.listdir(path_analyse) if not "." in m]
    n = len(models)

    y = np.zeros((n, len(tests)+2)) + np.inf
    e = np.zeros((n, len(tests)+2)) + np.inf
    x = np.zeros((n, len(tests)+2)).astype(str)
    x[:, :] = '---'

    pbar = tqdm(total=n)

    for i, m in enumerate(models):

        if "trainAtmo" in m : foldtest = "testAtmo1k"
        elif "train2Atmo" in m: foldtest = "test2Atmo1k"
        else : raise Exception("Models name as neither `trainAtmo` or `train2Atmo`")

        if foldtest not in os.listdir(f"{path_analyse}/{m}"): 

            print(f"The folder {foldtest} is not in {path_analyse}/{m}")

        elif "resume.json" not in os.listdir(f"{path_analyse}/{m}/{foldtest}"):

            print(f"The file resume.json not in {path_analyse}/{m}/{foldtest}")

        else:

            with open(f"./results/output_simu/{foldtest}/hist_params.json", 'r') as f:
                train_params = json.load(f)

            with open(f"{path_analyse}/{m}/{foldtest}/resume.json", 'r') as f:
                resume_i = json.load(f)

            y[i][-2] = 0.0

            for j, test in enumerate(tests):

                m, s = resume_i[test.lower()]
                MIN, MAX = train_params[f"ATM_{test.upper()}"]

                y[i][j] = m
                e[i][j] = s
                x[i][j] = f"{m:.3f} ~ {s:.3f}"

                y[i][-2] += m / (MAX - MIN) / 3 * 100

            x[i][-2] = f"{y[i][-2]:.2f} %"

        pbar.update()

    order = np.zeros(n)
    for m, cl in enumerate(np.argsort(y[:, -2])):
        order[cl] = m

    y[:, -1] = order
    x[:, -1] = order.astype(int).astype(str)
    e[:, -1] = 0.0





    for sorting, sorting_str in [(False, ""), (True, "_sorting")]:

        with open(f"{path_resume}/{score}{sorting_str}.html", "w") as f:

            html_codes = [f"<h1>Score {score}</h1>"]
            html_codes.append(generate_html_table(tests+["Final Score", "Classement"], models, x, y, sorting=sorting))
            f.write('\n'.join(html_codes))