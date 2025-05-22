from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

import coloralf as c
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import torch

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


    # Selection du device
    if "gpu" in sys.argv and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        if "gpu" in sys.argv : print(f"{c.r}WARNING : GPU is not available for torch ... device turn to CPU ... ")
        device = torch.device("cpu")
    print(f"{c.ly}INFO : Utilisation de l'appareil pour l'inference : {c.tu}{device}{c.d}{c.d}")


    # path
    path_model_state = f"./results/Spec2vecModels_Results/{model}/states/{train}_{lr}_best.pth"
    path_image = f"./results/output_simu/{test}/image/image_{image}.npy"
    path_save = f"./results/analyse/gradcam"
    os.makedirs(path_save, exist_ok=True)
    filename = f"{model}_{train}_{lr} - {test}_image_{image}"


    # Load model
    model = SCaM_Model()
    model.load_state_dict(torch.load(path_model_state, map_location=device)['model_state_dict'])
    model.eval()
    model.to(device)


    # Image: numpy array [128, 1024] -> torch tensor [1, 1, 128, 1024]
    image_brut = np.load(path_image)
    image_norm = (image_brut - image_brut.min()) / (image_brut.max() - image_brut.min() + 1e-8)
    image_tensor = torch.tensor(image_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)


    target_layers = [model.conv3]
    input_tensor = image_tensor # Create an input tensor image for your model..
    # Note: input_tensor can be a batch tensor with several images!

    # We have to specify the target we want to generate the CAM for.
    targets = [ClassifierOutputTarget(413)]

    # Construct the CAM object once, and then re-use it on many images.
    with GradCAM(model=model, target_layers=target_layers) as cam:
        # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        # In this example grayscale_cam has only one image in the batch:
        grayscale_cam = grayscale_cam[0, :]
        # visualization = show_cam_on_image(, grayscale_cam, use_rgb=True)
        # You can also get the model out
        model_outputs = cam.outputs

        print(grayscale_cam.shape)
        # print(model_outputs.shape)

    plt.figure(figsize=(16, 8))
    plt.imshow(np.log10(image_brut+1), cmap='gray')
    plt.imshow(grayscale_cam, cmap='jet', alpha=0.5)
    plt.title(f"Grad-CAM for {filename} on neuron n°{n}")
    plt.colorbar()
    plt.axis('off')
    plt.savefig(f"{path_save}/{filename}.png")
    plt.close()