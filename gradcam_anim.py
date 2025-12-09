import os, sys, shutil, pickle
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.animation as animation




def makeAnim(back, gcams, path_save, pred, true, x=np.arange(300, 1100, 1), save_file="anim", extension='mp4', title=None, 
	time=10.0, fps=60, sqrooting=False, log=False, cmap='jet'):

	nb_frame = len(gcams)
	x_frame = np.linspace(0, len(x)-1, nb_frame).astype(int)

	if time is not None:
		fps = nb_frame / time

	# fig, ax = plt.subplots()
	fig, ax = plt.subplots(2, 1)

	vmax = max([np.max(gcams[i][:, 128:]) for i in range(len(gcams))])
	b = ax[0].imshow(np.log10(back+1), cmap="gray")
	g = ax[0].imshow(gcams[0], cmap=cmap, vmax=vmax, alpha=0.5)

	ax_true, = ax[1].plot(x, true, color='g', label="pred")
	ax_pred, = ax[1].plot(x, pred, color='r', label="true")
	spot = ax[1].scatter([x[x_frame[0]]], [pred[x_frame[0]]], color='k', label="GradCam target", marker="*", zorder=9999)

	ax[0].axis('off')
	ax[1].legend()
	ax[1].set_xlabel(f'$\lambda$ (nm)')
	pbar = tqdm(total=len(gcams))


	def update(frame):
		pbar.update(1)

		# update image
		g.set_array(gcams[frame])

		# update pred target
		new_coords = np.array([[x[x_frame[frame]], pred[x_frame[frame]]]])
		spot.set_offsets(new_coords)
		
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

	makeAnim(data_gradcam["image_brut"], data_gradcam["gcams"], path_save=f"{path_gradcam}/{datafolder}", pred=data_gradcam["pred"], true=data_gradcam["true"])

	# data_gradcam = {
    #     "gcams" : gcams,
    #     "pred" : pred,
    #     "true" : true,
    #     "image_brut" : image_brut,
    # }
