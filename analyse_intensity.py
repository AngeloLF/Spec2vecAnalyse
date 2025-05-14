import matplotlib.pyplot as plt
import numpy as np
import os, sys
from random import shuffle
from time import time
from scipy.optimize import curve_fit as cf

def linear_func(x, a , b):
	return a + b * x

folder2ana = f"./results/output_simu/test1k"

images = [f"{folder2ana}/image/{i}" for i in sorted(os.listdir(f"{folder2ana}/image"))]
spectrums = [f"{folder2ana}/spectrum/{s}" for s in sorted(os.listdir(f"{folder2ana}/spectrum"))]

n = len(images)

int_i = np.zeros(n)
int_s = np.zeros(n)

ti, ts = 0, 0
t0i, t0s = None, None

smin, specmin = 0, None

for k, (i, s) in enumerate(zip(images, spectrums)):

	t0i = time()
	ima = np.load(i)[:, 128:]
	ti += time() - t0i

	t0s = time()
	spec = np.load(s)
	ts = time() - t0s

	int_i[k] = np.sum(ima)
	int_s[k] = np.sum(spec)

	if np.sum(spec) > smin: 
		smin = np.sum(spec)
		specmin = spec



As, Bs = np.zeros(100), np.zeros(100)

for i in range(100):

	index = np.random.randint(0, n, int(0.9*n))

	popt, pcov = cf(linear_func, int_s[index], int_i[index])
	As[i] = popt[0]
	Bs[i] = popt[1]



a, da = np.mean(As), np.std(As)
b, db = np.mean(Bs), np.std(Bs)

print(f"X0 : {a:.0f} ~ {da:.0f}")
print(f"Coef : {b:.1f} ~ {db:.1f}")

x = np.linspace(0, np.max(int_s), 1000)
y = linear_func(x, a, b)
ymin = linear_func(x, a, b-db)
ymax = linear_func(x, a, b+db)

plt.plot(int_s, int_i, marker='.', linestyle='', color='k', alpha=0.8)
plt.fill_between(x, ymin, ymax, color='r', alpha=0.5)
plt.plot(x, y, color='r')
plt.xlabel(f"Spectrum intensity")
plt.ylabel(f"Image intensity")
# plt.yscale('log')
# plt.xscale('log')
plt.show()



