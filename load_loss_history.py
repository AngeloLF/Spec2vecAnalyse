import numpy as np
import matplotlib.pyplot as plt
import os


path_s2v_results = f"./results/Spec2vecModels_Results"
folder_loss = "loss"

colors = ['g', 'r', 'b', 'k', 'm', 'y']

all_loss = dict()



for model in os.listdir(path_s2v_results):

	for folder_train in os.listdir(f"{path_s2v_results}/{model}/{folder_loss}"):

		name = f"{model}_{folder_train}"[:-4]
		file = f"{path_s2v_results}/{model}/{folder_loss}/{folder_train}"

		all_loss[name] = file


plt.figure(figsize=(16, 8))

for (name, file), color in zip(all_loss.items(), colors):

	train, valid = np.load(file)

	plt.plot(train, color=color, linestyle=':')
	plt.plot(valid, color=color, linestyle='-', label=name)

plt.legend()
plt.title(f"Evolution of loss for each models")
plt.xlabel(f"Epoch")
plt.ylabel(f"Loss")
plt.yscale(f"log")
plt.savefig(f"./results/analyse/loss_history.png")