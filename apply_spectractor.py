import matplotlib.pyplot as plt
import pickle, json, os, sys, shutil
from astropy.io import fits
import numpy as np
from scipy import interpolate
from time import time

sys.path.append("./Spectractor")
from spectractor.extractor import extractor
from spectractor.extractor.spectrum import Spectrum
from spectractor import parameters



def openTest(testname, pathtest="./results/output_simu", varfile="variable_params.pck", hpjson="hparams.json", histjson="hist_params.json", config = "./Spectractor/config/ctio.ini"):

    testdir = f"{pathtest}/{testname}"
    pred = f"pred_Spectractor_x_x_0e+00"

    with open(f"{testdir}/{hpjson}", "r") as fjson:
        hp = json.load(fjson)

    with open(f"{testdir}/{histjson}", "r") as fjson:
        hs = json.load(fjson)

    with open(f"{testdir}/{varfile}", "rb") as fpck:
        vp = pickle.load(fpck)

    images = np.sort([f"{testdir}/image/{fimage}" for fimage in os.listdir(f"{testdir}/image")])

    gain, ron = hp["CCD_GAIN"], hs["CCD_READ_OUT_NOISE"]
    
    header = fits.Header()
    header["DISPERSER"] = hp["DISPERSER"]
    header["EXPTIME"] = hs["EXPOSURE"]
    header['OUTTEMP'] = hs["ATM_TEMPERATURE"]
    header['OUTPRESS'] = hp["OBS_PRESSURE"]
    header['OUTHUM'] = hs["ATM_HUMIDITY"]
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

    times = np.zeros(len(images))

    for n in range(len(images)):

        header["TARGET"] = vp["TARGET"][n]
        header["AIRMASS"] = vp["ATM_AIRMASS"][n]
        header["ANGLE"] = - vp["ROTATION_ANGLE"][n]

        print(f"Angle : {header['ANGLE']}")
        print(vp.keys())
        
        data = np.zeros((hp["SIM_NX"], hp["SIM_NX"])) + hs["BACKGROUND_LEVEL"] * hs["EXPOSURE"]

        d = np.copy(data).astype(np.float32) * gain
        noisy = np.random.poisson(d).astype(np.float32)
        noisy += np.random.normal(scale=hs["CCD_READ_OUT_NOISE"]*np.ones_like(noisy)).astype(np.float32)

        data = noisy / gain
        min_noz = np.min(data[data > 0])
        data[data <= 0] = min_noz

        bande = int((hp["SIM_NX"] - hp["SIM_NY"])/2)
        data[bande:bande+hp["SIM_NY"], :] = np.load(images[n])[::-1, :]

        hdu = fits.PrimaryHDU(data=data.astype(np.float32), header=header)
        hdul = fits.HDUList([hdu])

        true_name = images[n].split("_")[-1][:-4]

        savefile = f"{testdir}/image_fits/images_{true_name}.fits"
        hdul.writeto(savefile, overwrite=True)
        print(f"Creation of {savefile}")

        t0 = time()

        spectrum = extractor.Spectractor(savefile, f"{testdir}/spectrum_fits", guess=[64, 512], target_label=vp["TARGET"][n], disperser_label=hp["DISPERSER"], config=config)
        spectrum.convert_from_flam_to_ADUrate()

        xp = spectrum.lambdas
        yp = spectrum.data * gain * spectrum.expo
        yt = np.load(f"{testdir}/spectrum/spectrum_{true_name}.npy")
        fact = np.max(yp)/np.max(yt)

        finterp = interpolate.interp1d(xp, yp, kind='linear', bounds_error=False, fill_value=0.0)
        ypi = finterp(xt) / fact

        times[n] = time() - t0

        np.save(f"{testdir}/{pred}/spectrum_{true_name}.npy", ypi)

    print(f"THE TIMES : {np.mean(times):.6f} += {np.std(times):.6f}")




if __name__ == "__main__":

    test_folder = sys.argv[1]
    openTest(test_folder)