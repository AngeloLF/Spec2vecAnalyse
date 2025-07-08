import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os, sys, shutil

sys.path.append('./Spec2vecModels/')
from get_argv import get_argv, get_device
from train_models import load_from_pretrained



def main():

    # ----- PARAMETRES -----
    output_neuron_index = 100
    

    # ----- CHARGER MODELE -----
    Args = get_argv(sys.argv[1:], prog="gradcam")
    device = get_device(Args)
    model, _ = load_from_pretrained(Args.model, Args.loss, Args.fulltrain_str, Args.lr_str, device=device)
    model.eval()


    # path
    path_image = f"./results/output_simu/{Args.test}/image/image_{Args.image}.npy"
    path_spectrum = f"./results/output_simu/{Args.test}/spectrum/spectrum_{Args.image}.npy"
    path_gradcam = f"./results/analyse/gradcam"
    os.makedirs(path_gradcam, exist_ok=True)

    path_save_image = f"{path_gradcam}"
    path_save_spectrum = f"{path_gradcam}/spectrum"
    os.makedirs(path_save_image, exist_ok=True)
    os.makedirs(path_save_spectrum, exist_ok=True)


    
    # ----- CHARGER IMAGE -----
    true = np.load(path_spectrum)
    image_np = np.load(path_image)  # shape attendue: (1,128,1024)
    image_tensor = torch.tensor(image_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device) # torch.from_numpy(image_np).unsqueeze(0).float().to(device)  # shape: (1,1,128,1024)
    image_tensor.requires_grad = True

    # ----- FORWARD -----
    output = model(image_tensor)  # shape: (1,800)
    pred = output.detach().numpy()[0]

    # ----- SELECTION DU NEURONE CIBLE -----
    neuron_output = output[0, output_neuron_index]

    # ----- BACKWARD POUR OBTENIR LE GRADIENT PAR RAPPORT A L'ENTREE -----
    model.zero_grad()
    neuron_output.backward()

    # ----- SALIENCY MAP -----
    saliency = image_tensor.grad.abs().detach().cpu().numpy()[0,0]  # shape: (128,1024)

    # ----- NORMALISATION POUR VISUALISATION -----
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)

    # ----- AFFICHAGE -----

    filename = f"{Args.fullname} - {Args.test}_image_{Args.image}"

    x = np.linspace(300, 1100, 800)
    plt.figure(figsize=(16, 8))
    plt.plot(x, true, c='g', label="True")
    plt.plot(x, pred, c='r', label="Pred")
    plt.legend()
    plt.savefig(f"{path_save_spectrum}/spectrum {filename}.png")
    if "show" in sys.argv : plt.show()
    plt.close()


    plt.figure(figsize=(12,6))
    plt.imshow(saliency, cmap="hot")
    plt.title(f"Saliency map for output neuron {output_neuron_index}")
    plt.colorbar()
    plt.show()

    # ----- SAUVEGARDE -----
    np.save(f"saliency_neuron_{output_neuron_index}.npy", saliency)
    print(f"Saliency map saved to saliency_neuron_{output_neuron_index}.npy")

if __name__ == "__main__":
    main()
