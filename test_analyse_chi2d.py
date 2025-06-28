import os, sys, json, pickle
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(f"./SpecSimulator")
from simulator import SpecSimulator
import utils_spec.psf_func as pf




if __name__ == "__main__":

    psf_function = {
        # For l lambdas in nm :
        # f : def func of (XX, YY, amplitude, x, y, f_argv[0](l, *argv[0]), ..., f_argv[n](l, *argv[n]))
        'f' : pf.moffat2d_jit,

        # function for argument
        'f_arg' : [pf.simpleLinear, pf.simpleLinear],
        
        # argument for argument function
        'arg' : [[3.0], [3.0]],

        # argument order 0
        'order0' : {'amplitude':22900.0, 'arg':[3.0, 2.0]},

        # timbre size function
        'timbre' : pf.moffat2d_timbre,
    }


    sim = SpecSimulator(psf_function=psf_function, savingFolders=False, target_set="setAll")

    if 1:
        pred = "pred_SCaM_chi2_train16k_5e-05"
        testfolder = "test1k"
        num_specs = ["0136"]

    if 0:
        pred = "pred_Spectractor_x_x_0e+00"
        testfolder = "test1k"
        num_specs = ["0044"]

    if 0:
        pred = "pred_Spectractor_x_x_0e+00"
        testfolder = "test4"
        num_specs = ["0"]

    testdir = f"./results/output_simu/{testfolder}"


    with open(f"{testdir}/hparams.json", "r") as fjson:
        hp = json.load(fjson)

    with open(f"{testdir}/hist_params.json", "r") as fjson:
        hs = json.load(fjson)

    with open(f"{testdir}/variable_params.pck", "rb") as fpck:
        vp = pickle.load(fpck)

    print(vp.keys())

    for num in num_specs:

        n = int(num)

        for param in vp.keys():

            # set var params
            if param[:4] != "arg.": 

                sim.__setattr__(param, vp[param][n])

            # set var arg psf
            else:

                sim.psf_function['arg'][0][0] = vp[param][n]

        arg_timbre = [int(np.round(np.max(f_arg(sim.lambdas, *arg)))) for f_arg, arg in zip(sim.psf_function['f_arg'], sim.psf_function['arg'])]
        timbre_size = sim.psf_function['timbre'](*arg_timbre)

        print(timbre_size)

        true = np.load(f"{testdir}/spectrum/spectrum_{num}.npy")
        pred = np.load(f"{testdir}/{pred}/spectrum_{num}.npy")

        plt.plot(true, c='g')
        plt.plot(pred, c='r')
        plt.show()

        true_simu = np.load(f"{testdir}/image/image_{num}.npy")
        pred_simu, _, xc, yc = sim.makeSim(num_simu=num, updateParams=False, giveSpectrum=pred, with_noise=False)

        print(true_simu.shape)

        # true_simu = true_simu[:, 128:]
        # pred_simu = pred_simu[:, 128:]
        mask = np.zeros_like(true_simu)
        for xi, yi in zip(xc, yc):
            mask[int(max(0, yi-timbre_size)):int(min(true_simu.shape[0], yi+timbre_size)), int(max(0, xi-timbre_size)):int(min(true_simu.shape[1], xi+timbre_size))] = 1
        true_simu[~(mask == 1)] = 0
        pred_simu[~(mask == 1)] = 0
        N = np.sum(mask)

        residus = true_simu - pred_simu
        vmax = max(np.abs(np.min(residus)), np.max(residus))

        sigma_CDD = hs["CCD_READ_OUT_NOISE"] / hp["CCD_GAIN"] 
        chi2eq = residus**2 / (sigma_CDD**2 + true_simu / hp["CCD_GAIN"]) * np.sign(residus)
        vmax2 = max(np.abs(np.min(chi2eq)), np.max(chi2eq))

        plt.subplot(211)
        plt.title(f"Resisus")
        plt.imshow(residus, cmap='bwr', vmin=-vmax/2, vmax=vmax/2)
        plt.colorbar()

        plt.subplot(212)
        title = "sign(res) \\cdot \\frac{res^2}{\\sigma^2_{Read} + it / \\sigma_{gain}}"
        plt.title(f"${title}$")
        plt.imshow(chi2eq, cmap='bwr', vmin=-vmax2/2, vmax=vmax2/2)
        plt.colorbar()

        plt.show()

        # N = np.shape(true_simu)[0] * np.shape(true_simu)[1]
        print(N, np.sum(np.abs(chi2eq)) / N)
