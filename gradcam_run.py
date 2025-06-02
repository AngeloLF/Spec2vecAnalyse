from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

import coloralf as c
import os, sys, shutil, pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
from time import time
from tqdm import tqdm

sys.path.append('./Spec2vecModels/')
from get_argv import get_argv, get_device
from train_models import load_from_pretrained






def gradcam_run(model, image, layers, neurons, aug_smooth=False):

    targets = [ClassifierOutputTarget(int(i)) for i in list(neurons)]
    if len(neurons) > 1 : pbar = tqdm(total=len(neurons))
    cams = list()

    with GradCAM(model=model, target_layers=target_layers) as cam:
        
        for neuron in neurons:
            targets = [ClassifierOutputTarget(neuron)]
            cam_result = cam(input_tensor=image, targets=targets, eigen_smooth=True, aug_smooth=aug_smooth)
            cams.append(cam_result[0, :])
            if len(neurons) > 1 : pbar.update(1)

        pred = cam.outputs.detach().numpy()[0]

    if len(neurons) > 1 : pbar.close()

    return cams, pred




def make_one(model, true, image_brut, image, layers, n, path_save_image, path_save_spectrum, filename, show=False, aug_smooth=False):


    gcams, pred = gradcam_run(model, image, layers, neurons=[n], aug_smooth=aug_smooth)
    gcam = gcams[0]

    vmax = np.max(gcam[:, 128:])

    plt.figure(figsize=(24, 8))
    plt.imshow(np.log10(image_brut+1), cmap='gray')
    plt.imshow(gcam, cmap='jet', alpha=0.5, vmax=vmax)
    plt.title(f"Grad-CAM for {filename} on neuron n°{n}")
    plt.axis('off')
    plt.savefig(f"{path_save_image}/{filename}.png")
    if show : plt.colorbar(); plt.show()
    plt.close()

    x = np.linspace(300, 1100, 800)
    plt.figure(figsize=(16, 8))
    plt.plot(x, true, c='g', label="True")
    plt.plot(x, pred, c='r', label="Pred")
    plt.legend()
    plt.savefig(f"{path_save_spectrum}/spectrum {filename}.png")
    if show : plt.show()
    plt.close()



def make_mult_and_save(model, true, image_brut, image, layers, nb_neurons, path_gradcam, filename, nneurons=800, aug_smooth=False):

    neurons = np.linspace(0, nneurons-1, nb_neurons).astype(int)

    path_save = f"{path_gradcam}/{filename}"
    if filename in os.listdir(path_gradcam) : shutil.rmtree(path_save)
    os.mkdir(path_save)

    gcams, pred = gradcam_run(model, image, layers, neurons=neurons, aug_smooth=aug_smooth)

    data_gradcam = {
        "gcams" : gcams,
        "pred" : pred,
        "true" : true,
        "image_brut" : image_brut,
    }

    with open(f"{path_save}/data_gradcam.pkl", "wb") as f : pickle.dump(data_gradcam, f)









if __name__ == "__main__":

    Args = get_argv(sys.argv[1:], prog="gradcam")
    device = get_device(Args)
    model, _ = load_from_pretrained(Args.model, Args.loss, Args.fulltrain_str, Args.lr_str, device=device)


    # path
    path_image = f"./results/output_simu/{Args.test}/image/image_{Args.image}.npy"
    path_spectrum = f"./results/output_simu/{Args.test}/spectrum/spectrum_{Args.image}.npy"
    path_gradcam = f"./results/analyse/gradcam"
    os.makedirs(path_gradcam, exist_ok=True)

    path_save_image = f"{path_gradcam}"
    path_save_spectrum = f"{path_gradcam}/spectrum"
    os.makedirs(path_save_image, exist_ok=True)
    os.makedirs(path_save_spectrum, exist_ok=True)

    # Load image
    true = np.load(path_spectrum)
    image_brut = np.load(path_image)
    if "no0" in sys.argv : image_brut[:, :128] = 0 
    if "noS" in sys.argv : image_brut[:, 500:600] = 0 
    image_tensor = torch.tensor(image_brut, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    # define target layers
    if Args.model == "SCaM" : target_layers = [model.conv3]
    else : raise Exception(f"{c.r}WARNING : target layers for model architecture {Args.model} not define{c.d}")


    if Args.mode == "one":

        filename = f"{Args.fullname} - {Args.test}_image_{Args.image}"
        make_one(model, true, image_brut, image_tensor, target_layers, Args.n, path_save_image, path_save_spectrum, filename, show=Args.show, aug_smooth=Args.aug_smooth)

    elif Args.mode == "mult":

        filename = f"{Args.fullname} - {Args.test}_image_{Args.image}"
        make_mult_and_save(model, true, image_brut, image_tensor, target_layers, Args.n, path_gradcam, filename, aug_smooth=Args.aug_smooth)

    