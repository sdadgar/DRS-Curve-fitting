import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp2d
from scipy.optimize import minimize
from mpl_toolkits.mplot3d import Axes3D
# Importing required libraroes

def calcZonios(InitVal, R_extracted, dHb, HbO2, mel, wavelength, mu_a_table, mu_sp_table, LUT):
    """
    calcZonios computes the fit based on extracted spectrum and input values

    Arguments:
    InitVal      -- A (1x5) array of initial values
    R_extracted  -- A (601x1) array is the spectrum that is going to be Fit
    dHb, HbO2, mel, wavelength, mu_a_table, mu_sp_table, LUT -- Are (601x1) arrays to be used in curve fitting algorithm

    Returns:
    chisq -- Chi-sqaure error between measured spectrum and the model fit
    """
    A = InitVal[0]
    B = InitVal[1]
    M = InitVal[2]
    C = InitVal[3]
    S = InitVal[4]
    # Unpacking initiall values

    mu_a  = 2.303 * C * (S * HbO2 + (1 - S) * dHb) + M * mel
    mu_sp = A * 10 * np.power(wavelength / 600, -B)
    # Calculating absorption and scattering of light based on initial values

    f = interp2d(mu_sp_table.T, mu_a_table, LUT, kind = 'linear')
    # Creating a 2-D interpolative model based on known absorption and scattering properties

    R_expected = np.array([])
    for i in range(len(mu_a)):
        int = f(mu_sp[i],mu_a[i])
        R_expected = np.append(R_expected, int)
    # Interpolating tumor values based on generated model

    chisq = np.sum(np.power((R_expected - R_extracted) / R_extracted, 2))
    # Calculating the chi-square error between the raw tumor spectrum and generated fit

    return chisq

def calcR(opt_vals, dHb, HbO2, mel, wavelength, mu_a_table, mu_sp_table, LUT):
    """
    calcR computes the final fit of the model to raw data based on optimized parameters

    Arguments:
    opt_vals      -- A (1x5) array of optimized values
    dHb, HbO2, mel, wavelength, mu_a_table, mu_sp_table, LUT -- Are (601x1) arrays to be used in curve fitting algorithm

    Returns:
    model_fit -- Chi-sqaure error between measured spectrum and the model fit
    """
    A = opt_vals[0]
    B = opt_vals[1]
    M = opt_vals[2]
    C = opt_vals[3]
    S = opt_vals[4]
    # Unpacking optimized values

    mu_a  = 2.303 * C * (S * HbO2 + (1 - S) * dHb) + M * mel
    mu_sp = A * 10 * np.power(wavelength / 600, -B)
    # Calculating absorption and scattering of light based on initial values

    f = interp2d(mu_sp_table.T, mu_a_table, LUT, kind='linear')
    # Creating a 2-D interpolative model based on known absorption and scattering properties

    model_fit = np.array([])
    for i in range(len(mu_a)):
        int = f(mu_sp[i],mu_a[i])
        model_fit = np.append(model_fit, int)
        # Interpolating tumor values based on generated model

    return model_fit

dir = r"C:\Users\krraz\Desktop\sd\tumor_data"
os.chdir(dir)
dirs = os.listdir(dir)
# Defining directory,changing it & getting a list of files in the directory

grads = []
for i in range(len(dirs)):
    intensity = pd.read_csv(dirs[i], header = None, usecols = [1])
    grads.append(intensity)
data = pd.concat(grads, axis = 1)
# Reading all of the csv files and concatenating them to one matrix

numerator   =  data.iloc[:, 0] - data.iloc[:, 1]
denominator = (data.iloc[:,-2] - data.iloc[:,-1]) * 0.8
# subtracting the background noise
reflectance = numerator / denominator
# calculating the reflectance according to equation of (A-Astd)/(std-stdbgd)

dir = r"C:\Users\krraz\Desktop\sd\LUT_files"
os.chdir(dir)
# chaning the directory to where the Lookup table is located at

dHb  = (pd.read_csv('eHb.csv', header = None, usecols = [1])).values
HbO2 = (pd.read_csv('eHbO2.csv', header = None, usecols = [1])).values
mel  = (pd.read_csv('emel.csv',  header = None, usecols = [1])).values
wavelength = (pd.read_csv('Wavelength.csv', header = None, usecols = [0])).values
mu_a_table  = (pd.read_csv('mua350.csv' , header = None, usecols = [0])).values
mu_sp_table = (pd.read_csv('musp_75.csv', header = None, usecols = [0])).values
LUT = (pd.read_csv('table2.csv', header = None)).values
# loading LUT files & Converting panda files to numpy arrays

SUP = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
# enable printing subscript & superscirpts

plt.plot(wavelength, HbO2, label = 'HbO2'.translate(SUB) )
plt.plot(wavelength, dHb, label = 'dHb' )
plt.plot(wavelength, mel, label = 'melanin' )
plt.xlabel('Wavelength (nm)')
plt.ylabel('Intensity')
plt.legend()
plt.show()
# plotting spectra of Hbo2, dHb, & mel

fig  = plt.figure()
ax   = fig.gca(projection = '3d')
surf = ax.plot_surface(mu_sp_table.T, mu_a_table, LUT)
ax.set_xlabel('Absorption (cm -1)'.translate(SUP))
ax.set_ylabel('Scattering (cm -1)'.translate(SUP))
ax.set_zlabel('Reflectance (AU)');
ax.view_init(elev = 30, azim = 140)
plt.show()
# plotting the LUT

args = (reflectance, dHb, HbO2, mel, wavelength, mu_a_table, mu_sp_table, LUT)
# Combining all of the LUT-related parameters into a tuple
InitVal  = np.array([0.75, 1.1, 1, 1, 0.85])
x_bounds = [[0.2, 1.5], [0, 1.4], [2, 3], [0.05, 7], [0, 1]]
# defining initial values to start & bounds

solution = minimize(calcZonios, InitVal, args = (reflectance, dHb, HbO2, mel, wavelength, mu_a_table, mu_sp_table, LUT), method = 'SLSQP', bounds = x_bounds, options = {'ftol': 1e-8, 'maxiter': 2000, 'disp': True})
fit = calcR(solution.x, dHb, HbO2, mel, wavelength, mu_a_table, mu_sp_table, LUT)

plt.plot(wavelength, reflectance, label = 'Measured' )
plt.plot(wavelength, fit, label = 'Fit' )
plt.xlabel('Wavelength (nm)')
plt.ylabel('Reflectance')
plt.legend()
plt.show()
