import os, sys, shutil, pickle
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap, Normalize
from scipy import ndimage
from scipy.ndimage import gaussian_filter



def gaussian_kernel(size=16, sigma=3.0):
    """Crée un noyau gaussien 2D normalisé."""
    x, y = np.meshgrid(np.linspace(-size//2 + 1, size//2, size), np.linspace(-size//2 + 1, size//2, size))
    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    return kernel / np.sum(kernel)



def transparent_red_cmap():

    # base_cmap = plt.cm.Reds
    # cmap_array = base_cmap(np.linspace(0, 1, 256))
    # cmap_array[:, -1] = np.linspace(0, 1, 256)

    rgba = np.ones((256,4))
    rgba[:,0] = 1.0   # R
    rgba[:,1] = 0.0   # G
    rgba[:,2] = 0.0   # B
    rgba[:,3] = np.linspace(0,1,256)
    
    cmap_transparent = LinearSegmentedColormap.from_list("transparent_red", rgba)

    return cmap_transparent



def makeAnim(back, gcams, path_save, preds, true, x=np.arange(300, 1100, 1), save_file="anim", extension='mp4', title=None, 
    time=10.0, fps=60, sqrooting=False, log=False, cmap='jet'):

    nb_frame = len(gcams)
    x_frame = np.linspace(0, len(x)-1, nb_frame).astype(int)

    if time is not None:
        fps = nb_frame / time

    # fig, ax = plt.subplots()
    fig, ax = plt.subplots(2, 1)
    cmap_custom = transparent_red_cmap()


    mean_visions = np.mean(gcams, axis=0)
    # vmax = max([np.max(gcams[i][:, 128:]) for i in range(len(gcams))])/10
    vmax = np.max(mean_visions)/2

    convoluted = gaussian_filter(gcams[0], sigma=(3.0, 3.0), mode='reflect')


    b = ax[0].imshow(np.log10(back+1), cmap="gray")
    g = ax[0].imshow(convoluted, cmap=cmap_custom, vmax=vmax)

    pred = np.mean(preds, axis=0)
    yerr = np.std(preds, axis=0)

    ax[1].plot(x, true, color='g', label="True")
    ax[1].errorbar(x, pred, yerr=yerr, c='r', label="Pred")
    spot = ax[1].scatter([x[x_frame[0]]], [pred[x_frame[0]]], color='k', label="Visions backward target", marker="*", zorder=9999)

    ax[0].axis('off')
    ax[1].legend()
    ax[1].set_xlabel(f'$\lambda$ (nm)')
    pbar = tqdm(total=len(gcams))


    def update(frame):
        pbar.update(1)

        # update image
        g.set_array(gaussian_filter(gcams[frame], sigma=(3.0, 3.0), mode='reflect'))
        # g.set_array(gcams[frame])

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

    path_visions = f"./results/analyse/visions"
    datafolder = sys.argv[1]

    with open(f"{path_visions}/{datafolder}/data_visions.pkl", "rb") as f : data_dict = pickle.load(f)

    makeAnim(data_dict["image_brut"], data_dict["visions"], path_save=f"{path_visions}/{datafolder}", preds=data_dict["pred"], true=data_dict["true"])

    # data_visions = {
    #     "visions" : visions,
    #     "pred" : preds,
    #     "true" : true,
    #     "image_brut" : image_brut,
    # }
