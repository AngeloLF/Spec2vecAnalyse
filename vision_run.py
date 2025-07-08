import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os, sys, shutil, pickle
from tqdm import tqdm

from captum.attr import IntegratedGradients, GuidedBackprop

sys.path.append('./Spec2vecModels/')
from get_argv import get_argv, get_device
from train_models import load_from_pretrained




def vision_run(model, image_brut, neurons, nneurons=800):

    visions = list()
    preds = np.zeros((len(neurons), nneurons))
    pbar = tqdm(total=len(neurons))

    for i, n in enumerate(neurons):

        image_tensor = torch.tensor(image_brut, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        image_tensor.requires_grad = True

        model.eval()
        output = model(image_tensor)
        preds[i] = output.detach().numpy()[0]
        
        neuron_output = output[0, n]
        model.zero_grad()
        neuron_output.backward()

        v = image_tensor.grad.abs().detach().cpu().numpy()[0, 0]  # shape: (128, 1024)
        if "norma" in sys.argv : v = (v - np.min(v)) / (np.max(v) - np.min(v) + 1e-8)

        visions.append(v)
        pbar.update(1)

    pbar.close()

    return preds, visions



def IG_run(model, image_brut, neurons, nneurons=800):

    visions = list()
    preds = np.zeros((len(neurons), nneurons))
    pbar = tqdm(total=len(neurons))

    image_tensor = torch.tensor(image_brut, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    for i, n in enumerate(neurons):

        model.eval()
        output = model(image_tensor)
        preds[i] = output.detach().numpy()[0]

        baseline = torch.zeros_like(image_tensor).to(device)

        ig = IntegratedGradients(model)

        attributions, delta = ig.attribute(
            inputs=image_tensor,
            baselines=baseline,
            target=int(n),
            return_convergence_delta=True,
            n_steps=200
        )
        
        neuron_output = output[0, n]
        model.zero_grad()
        neuron_output.backward()

        v = attributions.detach().cpu().numpy()[0,0]  # shape: (128,1024)
        v = np.abs(v)
        if "norma" in sys.argv : v = (v - np.min(v)) / (np.max(v) - np.min(v) + 1e-8)

        visions.append(v)
        pbar.update(1)

    pbar.close()

    return preds, visions



def guided_run(model, image_brut, neurons, nneurons=800):

    visions = list()
    preds = np.zeros((len(neurons), nneurons))
    pbar = tqdm(total=len(neurons))

    image_tensor = torch.tensor(image_brut, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    image_tensor.requires_grad = True

    for i, n in enumerate(neurons):

        model.eval()
        output = model(image_tensor)
        preds[i] = output.detach().numpy()[0]

        gbp = GuidedBackprop(model)

        attributions = gbp.attribute(
            inputs=image_tensor,
            target=int(n)
        )
        
        neuron_output = output[0, n]
        model.zero_grad()
        neuron_output.backward()

        v = attributions.detach().cpu().numpy()[0,0]  # shape: (128,1024)
        v = np.abs(v)
        if "norma" in sys.argv : v = (v - np.min(v)) / (np.max(v) - np.min(v) + 1e-8)

        visions.append(v)
        pbar.update(1)

    pbar.close()

    return preds, visions




def make_one(model, true, image_brut, n, path_save_image, path_save_spectrum, filename):

    if "ig" in sys.argv : preds, visions = IG_run(model, image_brut, neurons=[n])
    elif "g" in sys.argv : preds, visions = guided_run(model, image_brut, neurons=[n])
    else : preds, visions = vision_run(model, image_brut, neurons=[n])
    pred = preds[0]
    v = visions[0]

    vmax = np.max(v[:, 128:])

    plt.figure(figsize=(24, 8))
    plt.imshow(np.log10(image_brut+1), cmap='gray')
    plt.imshow(v, cmap='jet', alpha=0.5, vmax=vmax)
    plt.title(f"Vision backward for {filename} on neuron n°{n}")
    plt.axis('off')
    plt.savefig(f"{path_save_image}/{filename}.png")
    if "show" in sys.argv : plt.colorbar(); plt.show()
    plt.close()

    # x = np.linspace(300, 1100, 800)
    # plt.figure(figsize=(16, 8))
    # plt.plot(x, true, c='g', label="True")
    # plt.plot(x, pred, c='r', label="Pred")
    # plt.legend()
    # plt.savefig(f"{path_save_spectrum}/spectrum {filename}.png")
    # if "show" in sys.argv : plt.show()
    # plt.close()




def make_mult_and_save(model, true, image_brut, nb_neurons, path_visions, filename, nneurons=800):

    neurons = np.linspace(0, nneurons-1, nb_neurons).astype(int)

    path_save = f"{path_visions}/{filename}"
    if filename in os.listdir(path_visions) : shutil.rmtree(path_save)
    os.mkdir(path_save)

    if "ig" in sys.argv : preds, visions = IG_run(model, image_brut, neurons=neurons)
    elif "g" in sys.argv : preds, visions = guided_run(model, image_brut, neurons=neurons)
    else : preds, visions = vision_run(model, image_brut, neurons=neurons)

    if "show" in sys.argv:

        pred = np.mean(preds, axis=0)
        yerr = np.std(preds, axis=0)

        x = np.linspace(300, 1100, 800)
        plt.figure(figsize=(16, 8))
        plt.plot(x, true, c='g', label="True")
        plt.errorbar(x, pred, yerr=yerr, c='r', label="Pred")
        plt.legend()
        if "show" in sys.argv : plt.show()
        plt.close()


    data_dict = {
        "visions" : visions,
        "pred" : preds,
        "true" : true,
        "image_brut" : image_brut,
    }

    with open(f"{path_save}/data_visions.pkl", "wb") as f : pickle.dump(data_dict, f)




if __name__ == "__main__":

    Args = get_argv(sys.argv[1:], prog="gradcam")
    device = get_device(Args)
    model, _ = load_from_pretrained(Args.model, Args.loss, Args.fulltrain_str, Args.lr_str, device=device)
    model.eval()


    # path
    path_image = f"./results/output_simu/{Args.test}/image/image_{Args.image}.npy"
    path_spectrum = f"./results/output_simu/{Args.test}/spectrum/spectrum_{Args.image}.npy"
    path_visions = f"./results/analyse/visions"
    os.makedirs(path_visions, exist_ok=True)

    path_save_image = f"{path_visions}"
    path_save_spectrum = f"{path_visions}/spectrum"
    os.makedirs(path_save_image, exist_ok=True)
    os.makedirs(path_save_spectrum, exist_ok=True)

    true = np.load(path_spectrum)
    image_brut = np.load(path_image)


    if Args.mode == "one":

        filename = f"{Args.fullname} - {Args.test}_image_{Args.image}"
        make_one(model, true, image_brut, Args.n, path_save_image, path_save_spectrum, filename)

    elif Args.mode == "mult":

        filename = f"{Args.fullname} - {Args.test}_image_{Args.image}"
        make_mult_and_save(model, true, image_brut, Args.n, path_visions, filename)