import numpy as np
import matplotlib.pyplot as plt
import os, sys, pickle

sys.path.append('./Spec2vecModels/')
from get_argv import get_argv


tradargs = {
	"arg.0.0" : f"PSF Moffat $\gamma$",
}




if __name__ == "__main__":

	C = 12.
	gain = 3.

	### capture params
	Args = get_argv(sys.argv[1:], prog="residus")

	pathsave = f"./results/analyse/residus"
	os.makedirs(pathsave, exist_ok=True)


	pathdata = f"./results/output_simu/{Args.test}"
	predfolder = f"pred_{Args.fullname}"
	specfolder = Args.folder_output

	with open(f"{pathdata}/variable_params.pck", "rb") as f:
		varp = pickle.load(f)
	n = int(Args.spec)


	x = Args.wl

	yt = np.load(f"{pathdata}/{specfolder}/{Args.folder_output}_{Args.spec}.npy")
	yp = np.load(f"{pathdata}/{predfolder}/{Args.folder_output}_{Args.spec}.npy")

	res = yt - yp / np.sqrt(yt + C**2)

	fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(20, 8), gridspec_kw={'height_ratios': [3, 1]})

	ax1.set_title(f"For {Args.model_loss} train with {Args.fulltrain_str}_{Args.lr_str} | Flux : {np.sum(yt)/gain/1000:.0f} kADU")

	ax1.plot(x, yt, label='Spectrum to predict', color="g")
	ax1.plot(x, yp, label='Model prediction', color="r")

	ax1.scatter([], [], marker='d', label=f"Target : {varp['TARGET'][n]}", color='k')
	for key, val in varp.items():
		if key != "TARGET": 
			skey = key if key not in tradargs.keys() else tradargs[key]
			ax1.scatter([], [], marker='*', label=f"{skey} = {val[n]:.2f}", color='k')

	ax1.set_ylabel(f"{Args.test}/*/spectrum_{Args.spec}.npy")
	ax1.legend()

	ax2.axhline(0, color='k', linestyle='--', linewidth=1)
	ax2.errorbar(x, res, yerr=1, marker='.', linestyle='', color='k', linewidth=0.5)
	ax2.set_xlabel(f"$\lambda$ (nm)")
	ax2.set_ylabel("Residues")

	plt.tight_layout()
	if not Args.show:
		plt.savefig(f"{pathsave}/residus_{Args.fullname}_{Args.test}_spectrum{Args.spec}.png")
		plt.close()
	else:
		plt.show()