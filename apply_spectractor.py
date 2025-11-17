import matplotlib.pyplot as plt
import pickle, json, os, sys, shutil
from astropy.io import fits
import numpy as np
from scipy import interpolate
from time import time
import coloralf as c


spectractor_version = "Spectractor" 
for argv in sys.argv:
    if "=" in argv and argv.split("=")[0] == "specver":
        spectractor_version = argv.split("=")[1]
sys.path.append(f"./{spectractor_version}")
from spectractor.extractor import extractor
from spectractor.extractor.spectrum import Spectrum
from spectractor import parameters




def openTest(testname, pathtest="./results/output_simu", configPath = "config/ctio.ini", 
            makeonly=None, maxloop=1, specver="Spectractor", flipy=False):

    config = f"./{specver}/{configPath}"

    testdir = f"{pathtest}/{testname}"
    pred = f"pred_Spectractor_x_x_0e+00"

    if flipy : print(f"{c.g}INFO : Flipy True{c.d}")
    else : print(f"{c.y}INFO : Flipy False{c.d}")

    with open(f"{testdir}/hparams.json", "r") as fjson:
        hp = json.load(fjson)

    vp = np.load(f"{testdir}/vparams.npz")

    images = np.sort([f"{testdir}/image/{fimage}" for fimage in os.listdir(f"{testdir}/image")])

    gain, ron = hp["CCD_GAIN"], hp["cparams"]["CCD_READ_OUT_NOISE"]
    
    header = fits.Header()
    header["DISPERSER"] = hp["DISPERSER"]
    header["EXPTIME"] = hp["cparams"]["EXPOSURE"]
    header['OUTTEMP'] = hp["cparams"]["ATM_TEMPERATURE"]
    header['OUTPRESS'] = hp["OBS_PRESSURE"]
    header['OUTHUM'] = hp["cparams"]["ATM_HUMIDITY"]
    header['XLENGTH'] = hp["SIM_NX"]
    header['YLENGTH'] = hp["SIM_NX"]
    header['XPIXSIZE'] = hp["CCD_PIXEL2ARCSEC"]
    header['GTGAIN11'], header['GTGAIN12'], header['GTGAIN21'], header['GTGAIN22'] = gain, gain, gain, gain
    header['GTRON11'], header['GTRON12'], header['GTRON21'], header['GTRON22'] = ron, ron, ron, ron

    header["FILTERS"] = f"dia {hp['DISPERSER']}"
    header["FILTER1"] = "dia"
    header['FILTER2'] = hp["DISPERSER"]

    header["DATE-OBS"] = None
    header['RA'] = "12:53:8.10"
    header['DEC'] = "-18:33:27.78"
    header['HA'] = "01:52:42.92"

    xt = np.arange(hp["LAMBDA_MIN"], hp["LAMBDA_MAX"], hp["LAMBDA_STEP"])

    for fold in ["image_fits", "spectrum_fits", pred]:
        if fold in os.listdir(testdir) : shutil.rmtree(f"{testdir}/{fold}")
        os.mkdir(f"{testdir}/{fold}")

    times = np.zeros(hp["nsimu"])

    for n in range(hp["nsimu"]):

        if makeonly is None or int(makeonly) == n:

            header["TARGET"] = vp["TARGET"][n]
            header["AIRMASS"] = vp["ATM_AIRMASS"][n]
            header["ANGLE"] = vp["ROTATION_ANGLE"][n]
            if flipy : header["ANGLE"] *= -1

            print(f"\nMake {n}, with angle {header['ANGLE']}")
            
            data = np.zeros((hp["SIM_NX"], hp["SIM_NX"])) + hp["cparams"]["BACKGROUND_LEVEL"] * hp["cparams"]["EXPOSURE"]

            d = np.copy(data).astype(np.float32) * gain
            noisy = np.random.poisson(d).astype(np.float32)
            noisy += np.random.normal(scale=hp["cparams"]["CCD_READ_OUT_NOISE"]*np.ones_like(noisy)).astype(np.float32)

            data = noisy / gain
            min_noz = np.min(data[data > 0])
            data[data <= 0] = min_noz

            bande = int((hp["SIM_NX"] - hp["SIM_NY"])/2)

            if makeonly is None: 
                if flipy : data[bande:bande+hp["SIM_NY"], :] = np.load(images[n]) # [::-1, :]
                else : data[bande:bande+hp["SIM_NY"], :] = np.load(images[n])

                true_name = images[n].split("_")[-1][:-4]
            else: 
                if flipy : data[bande:bande+hp["SIM_NY"], :] = np.load(f"{testdir}/image/image_{makeonly}.npy") #[::-1, :]
                else : data[bande:bande+hp["SIM_NY"], :] = np.load(f"{testdir}/image/image_{makeonly}.npy")

                true_name = makeonly

            hdu = fits.PrimaryHDU(data=data.astype(np.float32), header=header)
            hdul = fits.HDUList([hdu])

            savefile = f"{testdir}/image_fits/images_{true_name}.fits"
            hdul.writeto(savefile, overwrite=True)
            print(f"Creation of {savefile}")

            t0 = time()

            if "debug" in sys.argv: 
                parameters.DEBUG = True 
                parameters.VERBOSE = True
            if "verb" in sys.argv:
                parameters.VERBOSE = True


            yt = np.load(f"{testdir}/spectrum/spectrum_{true_name}.npy")
            spectractor_ok = False
            loop = 0

            while not spectractor_ok:

                loop += 1

                try:
                    spectrum = extractor.Spectractor(savefile, f"{testdir}/spectrum_fits", guess=[64, 512], target_label=vp["TARGET"][n], disperser_label=hp["DISPERSER"], config=config)
                    spectrum.convert_from_flam_to_ADUrate()
                    xp = spectrum.lambdas
                    yp = spectrum.data * gain * spectrum.expo

                    fact = np.max(yp)/np.max(yt)
                    yp /= fact

                    finterp = interpolate.interp1d(xp, yp, kind='linear', bounds_error=False, fill_value=0.0)
                    ypi = finterp(xt)
                    spectractor_ok = True

                except Exception as e:
                    print(f"Exception : {e}")
                    print(f"{c.lr}WARNING : loop {loop}/{maxloop} extractor failed ...{c.d}")
                    if loop == maxloop:
                        spectractor_ok = True
                        ypi = np.zeros_like(xt) * np.nan
                        fact = 1.0
                        print(f"{c.r}Make nan yt ....{c.d}")

            times[n] = time() - t0

            np.save(f"{testdir}/{pred}/spectrum_{true_name}.npy", ypi)

            if "show" in sys.argv:
                plt.plot(xt, yt , c='g', label='Spectrum to predict')
                plt.plot(xt, ypi, c='r')
                plt.title(f"Mult : {fact}")
                plt.legend()
                plt.show()


            print(f"FACT : {fact}")

    print(f"THE TIMES : {np.mean(times):.6f} += {np.std(times):.6f} s")
    print(f"Total : {np.sum(times):.6f} s")




def viewResult(test_folder, name):

    xt = np.arange(300, 1100)
    yt = np.load(f"./results/output_simu/{test_folder}/spectrum/spectrum_{name}.npy")
    yp = np.load(f"./results/output_simu/{test_folder}/pred_Spectractor_x_x_0e+00/spectrum_{name}.npy")

    plt.plot(xt, yt , c='g', label='Spectrum to predict')
    plt.plot(xt, yp, c='r', label='prediction')
    plt.legend()
    plt.show()





if __name__ == "__main__":

    makeonly = None
    test_folder = sys.argv[1]

    flipy = False if "noflip" in sys.argv else True

    for arg in sys.argv[1:]:

        if arg[:9] == "makeonly=" : makeonly = arg[9:]


    if not "view" in sys.argv : openTest(test_folder, makeonly=makeonly, specver=spectractor_version, flipy=flipy)
    else : viewResult(test_folder, makeonly)

