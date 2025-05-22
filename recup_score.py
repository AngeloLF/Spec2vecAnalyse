import os, sys
import numpy as np

score_type = ["L1", "chi2"]
path_analyse = f"./results/analyse"
path_resume = f"{path_analyse}/all_resume"
os.makedirs(path_resume, exist_ok=True)



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

	if score in ["L1"]:

		y *= 100
		e *= 100

	print(y)
	print("*")
	print(e)

	with open(f"{path_resume}/{score}.tex", "w") as f:

		for i, typeScore in enumerate(["classic", "norma"]):

			f.write("\n\n\n\\begin{" + "table" + "}[H]\n")
			f.write("\\center\n")
			f.write("\\caption{"+ f"{score} with {typeScore} way" + "}\n")
			f.write("\\label{" + f"{typeScore}" + "}\n")

			f.write("\\begin{" + "tabular" + "}{" + f"l|{'c'*len(tests)}" + "}\n")
			f.write("\\hline\n")

			f.write(f"Model & {'&'.join(tests)} \\\\ \n")
			f.write("\\hline\n")

			for m, model in enumerate(models):

				lineLatex = [f"\\verb|{model}|"]

				for t, test in enumerate(tests):			

					if e[i, m, t] != 0 : lineLatex.append(f"${y[i, m, t]} \\pm {e[i, m, t]}$")
					else : lineLatex.append(f" ")

				f.write(' & '.join(lineLatex) + "\\\\ \n")

			f.write("\\end{" + "tabular" + "}\n")
			f.write("\\end{" + "table" + "}\n")





