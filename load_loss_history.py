import numpy as np
import matplotlib.pyplot as plt
import os


path_root = "./results"
path_savefig = "analyse"
if path_savefig not in os.listdir(path_root) : os.mkdir(f"{path_root}/{path_savefig}")
path_s2v_results = f"{path_root}/Spec2vecModels_Results"
folder_loss = "loss"

colors = ['g', 'r', 'b', 'k', 'm', 'y', 'darkred', 'darkgreen']

all_loss = dict()



for model in os.listdir(path_s2v_results):

	for folder_train in os.listdir(f"{path_s2v_results}/{model}/{folder_loss}"):

		name = f"{model}_{folder_train}"[:-4]
		file = f"{path_s2v_results}/{model}/{folder_loss}/{folder_train}"

		all_loss[name] = file

print(f"ALL loss : {all_loss}")
plt.figure(figsize=(16, 8))

for (name, file), color in zip(all_loss.items(), colors):

	train, valid = np.load(file)

	xmin = np.argmin(valid)
	ymin = np.min(valid)

	plt.plot(train, color=color, linestyle=':')
	plt.plot(valid, color=color, linestyle='-', label=f"{name} : best at epoch {xmin} = {ymin:.2f}")
	plt.scatter(xmin, ymin, facecolor=color, edgecolor='k', marker='d')

plt.legend()
plt.title(f"Evolution of loss for each models")
plt.xlabel(f"Epoch")
plt.ylabel(f"Loss")
plt.yscale(f"log")
plt.savefig(f"./{path_root}/{path_savefig}/loss_history.png")