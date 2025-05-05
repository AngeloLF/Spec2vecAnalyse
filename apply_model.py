# from model import SpectralModel, SpectralDataset
import torch

import json, pickle, sys, os, shutil
from tqdm import tqdm
import coloralf as c
import numpy as np

sys.path.append('./')

if __name__ == "__main__":

	model_name = None

	for argv in sys.argv[1:]:

		if argv[:6] == "model=" : model_name = argv[6:].lower()

	# Define model
	if model_name is None:
		print(f"{c.r}WARNING : model name is not define (python ... model=<model_name>){c.d}")
		sys.exit()

	# Define path and folders dirs
	path_results = './results'

	path_test = path_results
	folders_test_lsp = ["output_test_noisy"]
	folders_test_rdm = ["output_simu/test128"]

	path_train = f"{path_results}/output_simu"
	folder_train = "train256"

	name_of_image_folders = "image"
	name_of_spectrum_folders = "spectrum"


	# Selection du device
	if "gpu" in sys.argv and torch.cuda.is_available():
		device = torch.device("cuda")
	else:
		if "gpu" in sys.argv : print(f"{c.r}WARNING : GPU is not available for torch ... device turn to CPU ... ")
		device = torch.device("cpu")
	print(f"{c.ly}INFO : Utilisation de l'appareil pour l'inference : {c.tu}{device}{c.d}")


	if model_name in ["SCaM", "scam", "SCAM"]:

		from Spec2vecModels.SCaM.model import SCaM_Model as model
		from Spec2vecModels.SCaM.model import SCaM_Dataset as model_dataset
		model_name = "SCaM"

	elif model_name in ["JEC_Unet", "jec_unet", "JEC_UNET"]:

		from Spec2vecModels.JEC_Unet.model import UNet as model
		from Spec2vecModels.JEC_Unet.model import CustumDataset as model_dataset
		model_name = "JEC_Unet"

	else:

		print(f"{c.r}WARNING : model name {c.lr}{model_name}{c.r} unknow ...{c.d}")
		sys.exit()



	
	# Loading model
	print(f"{c.ly}INFO : Loading model {c.tu}{model_name}{c.ru} ...{c.d}")
	loaded_model = model()

	# Loading training
	MODEL_W = f"./results/Spec2vecModels_Results/{model_name}/states/{folder_train}.pth"
	print(f"{c.ly}INFO : Loading {loaded_model.nameOfThisModel} with file {c.tu}{MODEL_W}{c.ru} ... {c.d}")
	state = torch.load(MODEL_W)
	loaded_model.load_state_dict(state['model_state_dict'])
	loaded_model.eval()
	loaded_model.to(device)
	print(f"{c.ly}Loading ok{c.d}")
	
	pred_fold_name = f"pred_{loaded_model.nameOfThisModel}_{folder_train}"


	# search folder test for apply model
	folders2pred = [f"{path_results}/{fold}" for fold in folders_test_rdm]

	for ftest in folders_test_lsp:

		folders2pred += [f"{path_test}/{ftest}/{sf}" for sf in os.listdir(f"{path_test}/{ftest}") if "test" in sf]


	nb_folds = len(folders2pred)

	for i, fold in enumerate(folders2pred):

		print(f"{c.lg}Make {fold} [{i+1}/{nb_folds}] ...{c.d}")

		if pred_fold_name in os.listdir(fold) : shutil.rmtree(f"{fold}/{pred_fold_name}")
		os.mkdir(f"{fold}/{pred_fold_name}")
		print(f"{c.ly}INFO : creation of {fold}/{pred_fold_name}{c.d}")

		test_dataset = model_dataset(f"{fold}/{name_of_image_folders}", f"{fold}/{name_of_spectrum_folders}")

		with torch.no_grad():

			pbar = tqdm(total=len(test_dataset))

			for (img, true_tensor), spec_name in zip(test_dataset, test_dataset.spectrum_files):

				true_spec_name = spec_name.replace("\\", "/").split("/")[-1]

				test_image = img.unsqueeze(0).to(device)
				pred = loaded_model(test_image).cpu().numpy()[0]

				np.save(f"{fold}/{pred_fold_name}/{true_spec_name}", pred)
				pbar.update(1)

			pbar.close()



