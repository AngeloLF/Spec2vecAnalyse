# from model import SpectralModel, SpectralDataset
import torch

import json, pickle, sys, os, shutil
from tqdm import tqdm
import coloralf as c
import numpy as np

sys.path.append('./')

if __name__ == "__main__":

    model_name = None
    folder_train = None
    folder_test = None
    lr = None

    for argv in sys.argv[1:]:

        if argv[:6] == "model=" : model_name = argv[6:]
        if argv[:6] == "train=" : folder_train = argv[6:]
        if argv[:5] == "test=" : folder_test = argv[5:]
        if argv[:3] == "lr=" : lr = f"{float(argv[3:]):.0e}"

    # Define model
    if model_name is None:
        print(f"{c.r}WARNING : model name is not define (model=<model_name>){c.d}")
        raise Exception("Model name error")

    # Define train folder
    if folder_train is None:
        print(f"{c.r}WARNING : train folder is not define (train=<folder_train>){c.d}")
        raise Exception("Train folder error")

    # Define test folder
    if folder_test is None:
        print(f"{c.r}WARNING : test folder is not define (test=<folder_test>){c.d}")
        raise Exception("Test folder error")

    # Define lr
    if lr is None:
        print(f"{c.r}WARNING : learning rate is not define (lr=<lr>){c.d}")
        raise Exception("Lr error")


    path_results = './results'
    path_test = path_results
    path_train = f"{path_results}/output_simu"

    if "output" in folder_test : folders = [f"{path_test}/{folder_test}/{sf}" for sf in os.listdir(f"{path_test}/{folder_test}") if "test" in sf]
    else :                       folders = [f"{path_test}/output_simu/{folder_test}"]


    # Selection du device
    if "gpu" in sys.argv and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        if "gpu" in sys.argv : print(f"{c.r}WARNING : GPU is not available for torch ... device turn to CPU ... ")
        device = torch.device("cpu")
    print(f"{c.ly}INFO : Utilisation de l'appareil pour l'inference : {c.tu}{device}{c.d}{c.d}")


    if "SCaM_" in model_name:

        from Spec2vecModels.architecture_SCaM import SCaM_Model as model
        from Spec2vecModels.architecture_SCaM import SCaM_Dataset as model_dataset

    elif "SCaMv2_" in model_name:

        from Spec2vecModels.architecture_SCaMv2 import SCaMv2_Model as model
        from Spec2vecModels.architecture_SCaMv2 import SCaMv2_Dataset as model_dataset        

    elif "SotSu_" in model_name:

        from Spec2vecModels.architecture_SotSu import SotSu_Model as model
        from Spec2vecModels.architecture_SotSu import SotSu_Dataset as model_dataset

    elif "SotSuv2_" in model_name:

        from Spec2vecModels.architecture_SotSuv2 import SotSuv2_Model as model
        from Spec2vecModels.architecture_SotSuv2 import SotSuv2_Dataset as model_dataset

    else:

        print(f"{c.r}WARNING : model name {c.d}{c.lr}{model_name}{c.d}{c.r} unknow ...{c.d}")
        sys.exit()



    
    # Loading model
    print(f"{c.ly}INFO : Loading model {c.tu}{model_name}{c.ru} ...{c.d}")
    loaded_model = model()

    # Loading training
    MODEL_W = f"./results/Spec2vecModels_Results/{model_name}/states/{folder_train}_{lr}_best.pth"
    print(f"{c.ly}INFO : Loading {model_name} with file {c.tu}{MODEL_W}{c.ru} ... {c.d}")
    state = torch.load(MODEL_W, map_location=torch.device('cpu'))
    loaded_model.load_state_dict(state['model_state_dict'])
    loaded_model.eval()
    loaded_model.to(device)
    print(f"{c.ly}Loading ok{c.d}")
    
    pred_fold_name = f"pred_{model_name}_{folder_train}_{lr}"


    nb_folds = len(folders)

    for i, fold in enumerate(folders):

        print(f"{c.lg}Make {fold} [{i+1}/{nb_folds}] ...{c.d}")

        if pred_fold_name in os.listdir(fold) : shutil.rmtree(f"{fold}/{pred_fold_name}")
        os.mkdir(f"{fold}/{pred_fold_name}")
        print(f"{c.ly}INFO : creation of {fold}/{pred_fold_name}{c.d}")

        test_dataset = model_dataset(f"{fold}/image", f"{fold}/spectrum")

        with torch.no_grad():

            pbar = tqdm(total=len(test_dataset))

            for (img, true_tensor), spec_name in zip(test_dataset, test_dataset.spectrum_files):

                true_spec_name = spec_name.replace("\\", "/").split("/")[-1]

                test_image = img.unsqueeze(0).to(device)
                pred = loaded_model(test_image).cpu().numpy()[0]

                np.save(f"{fold}/{pred_fold_name}/{true_spec_name}", pred)
                pbar.update(1)

            pbar.close()



