import os, sys
import numpy as np

score_type = ["L1", "chi2"]
path_analyse = f"./results/analyse"
path_resume = f"{path_analyse}/all_resume"
os.makedirs(path_resume, exist_ok=True)



import numpy as np

def generate_html_table(colonnes, lignes, y):

    if y.shape != (len(lignes), len(colonnes)):
        raise ValueError("Les dimensions de y ne correspondent pas aux longueurs des listes ligne et colonne.")

    html = '<table border="1" style="border-collapse: collapse; text-align: center;">\n'
    
    # En-tête
    html += '  <tr><th></th>'  # Coin supérieur gauche vide
    for col in colonnes:
        html += f'<th>{col}</th>'
    html += '</tr>\n'
    
    # Lignes de données
    for i, ligne in enumerate(lignes):
        html += f'  <tr><th>{ligne}</th>'
        for j in range(len(colonnes)):
            html += f'<td>{y[i, j]}</td>'
        html += '</tr>\n'
    
    html += '</table>'
    return html





for score in score_type:

	print(f"\nConstruct score {score}")

	models = os.listdir(f"{path_analyse}/{score}")
	tests = list()

	for model in models:
		tests += os.listdir(f"{path_analyse}/{score}/{model}")

	tests = list(set(tests))

	print(tests)


	# Sorting lists
	models.sort()
	tests.sort()


	y = np.zeros((2, len(models), len(tests)))
	e = np.zeros((2, len(models), len(tests)))
	x = np.zeros((2, len(models), len(tests))).astype(str)

	
	for m, model in enumerate(models):

		for t, test in enumerate(tests):

			if test in os.listdir(f"{path_analyse}/{score}/{model}"):

				with open(f"{path_analyse}/{score}/{model}/{test}/resume.txt", "r") as f:
					data = f.read().split("\n")[:-1]

				for i, line in enumerate(data):

					label, score_i = line.split("=")
					mean, std = score_i.split("~")
					y[i, m, t] = mean
					e[i, m, t] = std
					x[i, m, t] = f"{mean} ~ {std}"


	if score in ["L1"]:

		y *= 100
		e *= 100

	print(y)
	print("*")
	print(e)

	print()

	with open(f"{path_resume}/{score}.html", "w") as f:

		html_codes = [f"<h1>Score {score}</h1>"]

		for i, typeScore in enumerate(["classic", "norma"]):

			html_codes.append(f"<h2>{typeScore}</h2>")
			html_codes.append(generate_html_table(tests, models, x[i]))

		f.write('\n'.join(html_codes))


	# 		f.write("\n\n\n\\begin{" + "table" + "}[H]\n")
	# 		f.write("\\center\n")
	# 		f.write("\\caption{"+ f"{score} with {typeScore} way" + "}\n")
	# 		f.write("\\label{" + f"{typeScore}" + "}\n")

	# 		f.write("\\begin{" + "tabular" + "}{" + f"l|{'c'*len(tests)}" + "}\n")
	# 		f.write("\\hline\n")

	# 		f.write(f"Model & {'&'.join(tests)} \\\\ \n")
	# 		f.write("\\hline\n")

	# 		for m, model in enumerate(models):

	# 			lineLatex = [f"\\verb|{model}|"]

	# 			for t, test in enumerate(tests):			

	# 				if e[i, m, t] != 0 : lineLatex.append(f"${y[i, m, t]} \\pm {e[i, m, t]}$")
	# 				else : lineLatex.append(f" ")

	# 			f.write(' & '.join(lineLatex) + "\\\\ \n")

	# 		f.write("\\end{" + "tabular" + "}\n")
	# 		f.write("\\end{" + "table" + "}\n")





