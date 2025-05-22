import os, sys, shutil, pickle
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.animation as animation




def makeAnim(back, gcams, path_save, save_file="anim", extension='mp4', title=None, 
	time=10.0, fps=60, sqrooting=False, log=False, cmap='jet'):

	if time is not None:
		fps = len(gcams) / time

	fig, ax = plt.subplots()
	ax.axis('off')

	vmax = max([np.max(gcams[i][:, 128:]) for i in range(len(gcams))])
	b = ax.imshow(np.log10(back+1), cmap="gray")
	g = ax.imshow(gcams[0], cmap=cmap, vmax=vmax, alpha=0.5)

	pbar = tqdm(total=len(gcams))


	def update(frame):
		pbar.update(1)
		g.set_array(gcams[frame])
		return g,

	ani = animation.FuncAnimation(fig, update, frames=len(gcams), blit=False, repeat=True)

	if title is not None : plt.title(title)
	# if folder not in os.listdir() : os.mkdir(folder)
	ani.save(f"{path_save}/{save_file}.{extension}", fps=fps, dpi=300)
	plt.close()

	pbar.close()



if __name__ == "__main__":

	path_gradcam = f"./results/analyse/gradcam"
	datafolder = sys.argv[1]

	with open(f"{path_gradcam}/{datafolder}/data_gradcam.pkl", "rb") as f : data_gradcam = pickle.load(f)

	# data_gradcam = {
    #     "gcams" : gcams,
    #     "pred" : pred,
    #     "true" : true,
    #     "image_brut" : image_brut,
    # }

	makeAnim(data_gradcam["image_brut"], data_gradcam["gcams"], path_save=f"{path_gradcam}/{datafolder}")
