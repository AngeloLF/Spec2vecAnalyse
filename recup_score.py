import os, sys
import numpy as np

score_type = ["L1", "chi2"]
path_analyse = f"./results/analyse"
path_resume = f"{path_analyse}/all_resume"
os.makedirs(path_resume, exist_ok=True)



import numpy as np

def generate_html_table(colonnes, lignes, text, y):


    tds = {
        "def" : "td",
            
        "far_min" : 'td style="background-color: #CCFFCC;"',
        "near_min" : 'td style="background-color: #66FF66;"',
        "min" : 'td style="background-color: #00CC00; font-weight: bold;"',

        "far_max" : 'td style="background-color: #FFCCCC;"',
        "near_max" : 'td style="background-color: #FF6666;"',
        "max" : 'td style="background-color: #CC0000; font-weight: bold;"',
    }

    if text.shape != (len(lignes), len(colonnes)):
        raise ValueError("Les dimensions de y ne correspondent pas aux longueurs des listes ligne et colonne.")

    html = '<table border="1" style="border-collapse: collapse; text-align: center;">\n'
    
    # En-tête
    html += '  <tr><th></th>'  # Coin supérieur gauche vide
    for col in colonnes:
        html += f'<th>{col}</th>'
    html += '</tr>\n'

    argmin, argmax = np.argmin(y, axis=0), np.argmax(y, axis=0)
    valmin, valmax = np.min(y, axis=0),    np.max(y, axis=0)

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
            html += f'<{td}>{text[i, j]}</td>'
        html += '</tr>\n'
    
    html += '</table>'
    return html





for score in score_type:

    models = os.listdir(f"{path_analyse}/{score}")
    tests = list()

    for model in models:
        tests += os.listdir(f"{path_analyse}/{score}/{model}")

    tests = list(set(tests))


    # Sorting lists
    models.sort()
    tests.sort()


    y = np.zeros((2, len(models), len(tests)+1)) + np.inf
    e = np.zeros((2, len(models), len(tests)+1)) + np.inf
    x = np.zeros((2, len(models), len(tests)+1)).astype(str)
    x[:, :] = '---'

    
    for m, model in enumerate(models):

        for t, test in enumerate(tests):

            tot_mean = [list(), list()]
            tot_std = [list(), list()]

            if test in os.listdir(f"{path_analyse}/{score}/{model}"):

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


            for i in range(2):
                mom = np.mean(tot_mean[i])
                soa = np.sum(np.array(tot_std[i])**2)**0.5
                y[i, m, -1] = mom
                e[i, m, -1] = soa
                x[i, m, -1] = f"{mom:.2f} ~ {soa:.2f}"

    





    with open(f"{path_resume}/{score}.html", "w") as f:

        html_codes = [f"<h1>Score {score}</h1>"]

        for i, typeScore in enumerate(["classic", "norma"]):

            html_codes.append(f"<h2>{typeScore}</h2>")
            html_codes.append(generate_html_table(tests+["Total"], models, x[i], y[i]))

        f.write('\n'.join(html_codes))


    #         f.write("\n\n\n\\begin{" + "table" + "}[H]\n")
    #         f.write("\\center\n")
    #         f.write("\\caption{"+ f"{score} with {typeScore} way" + "}\n")
    #         f.write("\\label{" + f"{typeScore}" + "}\n")

    #         f.write("\\begin{" + "tabular" + "}{" + f"l|{'c'*len(tests)}" + "}\n")
    #         f.write("\\hline\n")

    #         f.write(f"Model & {'&'.join(tests)} \\\\ \n")
    #         f.write("\\hline\n")

    #         for m, model in enumerate(models):

    #             lineLatex = [f"\\verb|{model}|"]

    #             for t, test in enumerate(tests):            

    #                 if e[i, m, t] != 0 : lineLatex.append(f"${y[i, m, t]} \\pm {e[i, m, t]}$")
    #                 else : lineLatex.append(f" ")

    #             f.write(' & '.join(lineLatex) + "\\\\ \n")

    #         f.write("\\end{" + "tabular" + "}\n")
    #         f.write("\\end{" + "table" + "}\n")





