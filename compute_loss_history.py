import numpy as np
import matplotlib.pyplot as plt
import os, sys, shutil






def work_on(lock=''):

	path_of_npy = f"{path_root}/Spec2vecModels_Results"
	colors = ['g', 'r', 'b', 'k', 'm', 'y', 'c', 'darkgreen', 'darkred', 'darkblue', 'gray', 'darkmagenta', 'orange', 'darkcyan']
	all_loss = dict()


	for fold in folders_loss:

		for model in os.listdir(path_of_npy):

			for filename in os.listdir(f"{path_of_npy}/{model}/{fold}"):

				name = f"{model}_{filename}"[:-4]

				if lock in name:
					
					file = f"{path_of_npy}/{model}/{fold}/{filename}"
					all_loss[name] = file

		print(f"Loss name for {fold} : {', '.join(list(all_loss.keys()))}")
		
		if fold != 'loss':

			plt.figure(figsize=(16, 8))

			for (name, file), color in zip(all_loss.items(), colors):

				train, valid = np.load(file)

				xmin = np.argmin(valid)
				ymin = np.min(valid)

				plt.plot(train, color=color, linestyle=':')
				plt.plot(valid, color=color, linestyle='-', label=f"{name} : best at epoch {xmin} = {ymin:.2f}")
				plt.scatter(xmin, ymin, facecolor=color, edgecolor='k', marker='d')

			plt.legend()
			plt.title(f"Evolution of {fold} for each models")
			plt.xlabel(f"Epoch")
			plt.ylabel(f"Loss")
			if "nolog" in sys.argv : plt.yscale(f"log")
			plt.savefig(f"./{path_losses}/{fold}_{lock}.png")

		else:

			for (name, file), color in zip(all_loss.items(), colors):

				plt.figure(figsize=(16, 8))

				train, valid = np.load(file)

				xmin = np.argmin(valid)
				ymin = np.min(valid)

				plt.plot(train, color=color, linestyle=':')
				plt.plot(valid, color=color, linestyle='-', label=f"{name} : best at epoch {xmin} = {ymin:.2f}")
				plt.scatter(xmin, ymin, facecolor=color, edgecolor='k', marker='d')

				plt.legend()
				plt.title(f"Evolution of loss for {name}")
				plt.xlabel(f"Epoch")
				plt.ylabel(f"Loss")
				if "nolog" in sys.argv : plt.yscale(f"log")
				plt.savefig(f"./{path_losses}/{fold}/loss_{name}_{lock}.png")




if __name__ == "__main__":

	path_root = "./results"

	dir_savefig = "analyse"
	path_savefig = f"{path_root}/{dir_savefig}"
	if dir_savefig not in os.listdir(path_root) : os.mkdir(path_savefig)

	dir_losses = "losses"
	path_losses = f"{path_savefig}/{dir_losses}"
	if dir_losses in os.listdir(path_savefig) : shutil.rmtree(path_losses)
	os.makedirs(path_losses, exist_ok=True)

	folders_loss = ["loss", "loss_mse", "loss_chi2"]
	if folders_loss[0] not in os.listdir(path_losses) : os.mkdir(f"{path_losses}/{folders_loss[0]}")


	if len(sys.argv) > 1:

		locks = sys.argv[1].split(",")

		for lock in locks :
			print(f"Make lock : {lock}")
			work_on(lock)

	else:

		work_on()