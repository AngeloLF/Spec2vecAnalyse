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

sys.path.append("./")
from Spec2vecModels.architecture_SCaM import SCaM_Model






def gradcam_run(model, image, layers, neurons):

    targets = [ClassifierOutputTarget(int(i)) for i in list(neurons)]
    if len(neurons) > 1 : pbar = tqdm(total=len(neurons))
    cams = list()

    with GradCAM(model=model, target_layers=target_layers) as cam:
        
        for neuron in neurons:
            targets = [ClassifierOutputTarget(neuron)]
            cam_result = cam(input_tensor=image, targets=targets, eigen_smooth=True)
            cams.append(cam_result[0, :])
            if len(neurons) > 1 : pbar.update(1)

        pred = cam.outputs.detach().numpy()[0]

    if len(neurons) > 1 : pbar.close()

    return cams, pred




def make_one(model, true, image_brut, image, layers, n, path_save_image, path_save_spectrum, filename):


    gcams, pred = gradcam_run(model, image, layers, neurons=[n])
    gcam = gcams[0]

    vmax = np.max(gcam[:, 128:])

    plt.figure(figsize=(24, 8))
    plt.imshow(np.log10(image_brut+1), cmap='gray')
    plt.imshow(gcam, cmap='jet', alpha=0.5, vmax=vmax)
    plt.title(f"Grad-CAM for {filename} on neuron n°{n}")
    plt.axis('off')
    plt.savefig(f"{path_save_image}/{filename}.png")
    plt.close()

    x = np.linspace(300, 1100, 800)
    plt.figure(figsize=(16, 8))
    plt.plot(x, true, c='g', label="True")
    plt.plot(x, pred, c='r', label="Pred")
    plt.legend()
    plt.savefig(f"{path_save_spectrum}/spectrum {filename}.png")
    plt.close()



def make_mult_and_save(model, true, image_brut, image, layers, nb_neurons, path_gradcam, filename, nneurons=800):

    neurons = np.linspace(0, nneurons-1, nb_neurons).astype(int)

    path_save = f"{path_gradcam}/{filename}"
    if filename in os.listdir(path_gradcam) : shutil.rmtree(path_save)
    os.mkdir(path_save)

    gcams, pred = gradcam_run(model, image, layers, neurons=neurons)

    data_gradcam = {
        "gcams" : gcams,
        "pred" : pred,
        "true" : true,
        "image_brut" : image_brut,
    }

    with open(f"{path_save}/data_gradcam.pkl", "wb") as f : pickle.dump(data_gradcam, f)









if __name__ == "__main__":

    model_name = None
    train = None
    test = None
    lr = None
    image = None
    n = None
    n_bins = 800

    for argv in sys.argv[1:]:

        if argv[:6] == "model=" : model_name = argv[6:]
        if argv[:6] == "train=" : train = argv[6:]
        if argv[:5] == "test=" : test = argv[5:]
        if argv[:3] == "lr=" : lr = f"{float(argv[3:]):.0e}"
        if argv[:6] == "image=" : image = argv[6:]
        if argv[:2] == "n=" : n = int(argv[2:])


    # Selection du device
    if "gpu" in sys.argv and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        if "gpu" in sys.argv : print(f"{c.r}WARNING : GPU is not available for torch ... device turn to CPU ... ")
        device = torch.device("cpu")
    print(f"{c.ly}INFO : Utilisation de l'appareil pour l'inference : {c.tu}{device}{c.d}{c.d}")


    # path
    path_model_state = f"./results/Spec2vecModels_Results/{model_name}/states/{train}_{lr}_best.pth"
    path_image = f"./results/output_simu/{test}/image/image_{image}.npy"
    path_spectrum = f"./results/output_simu/{test}/spectrum/spectrum_{image}.npy"
    path_gradcam = f"./results/analyse/gradcam"
    os.makedirs(path_gradcam, exist_ok=True)

    path_save_image = f"{path_gradcam}"
    path_save_spectrum = f"{path_gradcam}/spectrum"
    os.makedirs(path_save_image, exist_ok=True)
    os.makedirs(path_save_spectrum, exist_ok=True)


    # Load model
    model = SCaM_Model()
    model.load_state_dict(torch.load(path_model_state, map_location=device)['model_state_dict'])
    model.eval()
    model.to(device)

    # Load image
    true = np.load(path_spectrum)
    image_brut = np.load(path_image)
    image_tensor = torch.tensor(image_brut, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    # define target layers
    target_layers = [model.conv3]


    if "one" in sys.argv[1:]:

        filename = f"{model_name}_{train}_{lr} - {test}_image_{image}"
        make_one(model, true, image_brut, image_tensor, target_layers, n, path_save_image, path_save_spectrum, filename)

    elif "mult" in sys.argv[1:]:

        filename = f"{model_name}_{train}_{lr} - {test}"
        make_mult_and_save(model, true, image_brut, image_tensor, target_layers, n, path_gradcam, filename)

    