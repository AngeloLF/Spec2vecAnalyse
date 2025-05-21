import os, sys
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import coloralf as c

sys.path.append("./")
from Spec2vecModels.architecture_SCaM import SCaM_Model



if __name__ == "__main__":

    model = None
    train = None
    test = None
    lr = None
    image = None
    n = 413
    n_bins = 800

    for argv in sys.argv[1:]:

        if argv[:6] == "model=" : model = argv[6:]
        if argv[:6] == "train=" : train = argv[6:]
        if argv[:5] == "test=" : test = argv[5:]
        if argv[:3] == "lr=" : lr = f"{float(argv[3:]):.0e}"
        if argv[:6] == "image=" : image = argv[6:]
        if argv[:2] == "n=" : n = int(argv[2:])


    # path
    path_model_state = f"./results/Spec2vecModels_Results/{model}/states/{train}_{lr}_best.pth"
    path_image = f"./results/output_simu/{test}/image/image_{image}.npy"
    path_save = f"./results/analyse/gradcam"
    os.makedirs(path_save, exist_ok=True)
    filename = f"{model}_{train}_{lr} - {test}_image_{image}"


    # Load model
    model = SCaM_Model()
    model.load_state_dict(torch.load(path_model_state)['model_state_dict'])
    model.eval()

    # Image: numpy array [128, 1024] -> torch tensor [1, 1, 128, 1024]
    image_brut = np.load(path_image)
    image_norm = (image_brut - image_brut.min()) / (image_brut.max() - image_brut.min() + 1e-8)
    image_tensor = torch.tensor(image_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    target_layer_name='conv3'
    target_neuron_idx = n

    activations = []
    gradients = []

    # Hook pour forward
    def forward_hook(module, input, output):
        activations.append(output)

    # Hook pour backward
    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    # Récupère la couche cible
    target_layer = dict(model.named_modules())[target_layer_name]
    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)

    # Forward
    output = model(image_tensor)
    score = output[0, target_neuron_idx]

    # Backward
    model.zero_grad()
    score.backward()

    # Calcul du Grad-CAM
    grad = gradients[0]  # [1, C, H, W]
    act = activations[0] # [1, C, H, W]

    weights = grad.mean(dim=(2, 3), keepdim=True)  # Moyenne spatiale
    cam = F.relu((weights * act).sum(dim=1, keepdim=True))  # Pondération + ReLU

    # Resize vers la taille d'entrée
    cam = F.interpolate(cam, size=image_tensor.shape[2:], mode='bilinear', align_corners=False)
    cam = cam[0, 0].detach().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-6)  # Normalisation

    forward_handle.remove()
    backward_handle.remove()

    plt.figure(figsize=(16, 8))
    plt.imshow(np.log10(image_brut+1), cmap='gray')
    plt.imshow(cam, cmap='jet', alpha=0.5)
    plt.title(f"Grad-CAM for {filename} on neuron n°{n}")
    plt.colorbar()
    plt.axis('off')
    plt.savefig(f"{path_save}/{filename}.png")
    plt.close()
