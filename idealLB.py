# Ideal Link Budget (idealLB)
# Crispin Corpuz 
# University of Kansas
# Created 11/16

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# from fdtd import *

# Definitions
eps0 = 8.854e-12 # F/m
mu0 = 4e-7 * np.pi # H/m

# Input list of tissues
# e.g. ['Muscle', 'Fat', 'Bone'] or ['Muscle']
# Output dictionary of tissue data
def get_tissue_data_raw(tissue):
    csv_path = 'C:\\TLB-2\\cole_cole_tissues_gabriel_et_al.csv'
    full_data = pd.read_csv(csv_path,header=None,index_col=0).apply(pd.to_numeric, errors='coerce').to_dict('index')
    try:
        tissue_data_raw = full_data[tissue]
        tissue_data_raw[3] *= 1e-12
        tissue_data_raw[6] *= 1e-9
        tissue_data_raw[9] *= 1e-6
        tissue_data_raw[12] *= 1e-3
    except KeyError:
        print(f'Material "{tissue}" not found in database.')

    return tissue_data_raw

# Input raw tissue data list and frequency range
# Output complex permittivity over frequency range
def dielectric_model_4_cole_cole(tissue_data_list, frequency):
    # tissue_data_list = list(tissue_data_dict.values()) # leftover from when it was a dictionary
    # print(tissue_data_list)
    angular_frequency = 2 * np.pi * frequency
    e_inf = tissue_data_list[1]
    e_sum = 0
    for n in range(4): # 4 parameter Cole-Cole model
        e_sum += tissue_data_list[2 + 3 * n] / (1 + (1j * angular_frequency * tissue_data_list[3 + 3 * n]) ** (1 - tissue_data_list[4 + 3 * n]))
    stat_cond = tissue_data_list[14] / (1j * angular_frequency * eps0)
    e_complex = e_inf + e_sum + stat_cond
    real_er = np.real(e_complex)
    cond = -np.imag(e_complex) * (angular_frequency * eps0)
    return {'er': real_er, 'cond': cond}

def calc_gamma(dielectric_model_data, frequencies):
    angular_frequency = 2 * np.pi * frequencies
    e = dielectric_model_data['er'] * eps0
    s = dielectric_model_data['cond']
    mu = mu0 # assuming non-magnetic materials
    alpha = angular_frequency*np.sqrt(e*mu)*np.sqrt(0.5*(np.sqrt(1+(s/(angular_frequency*e))**2)-1))
    beta  = angular_frequency*np.sqrt(e*mu)*np.sqrt(0.5*(np.sqrt(1+(s/(angular_frequency*e))**2)+1))
    gamma = alpha + 1j*beta
    return gamma

def calc_impedance(dielectric_model_data, frequencies):
    angular_frequency = 2 * np.pi * frequencies
    e = dielectric_model_data['er'] * eps0
    s = dielectric_model_data['cond']
    mu = mu0 # assuming non-magnetic materials
    impedance = np.sqrt(1j*angular_frequency * mu/(s + 1j*angular_frequency*e))
    return impedance

def recursive_ref_coeff(boundary_index, thicknesses, gammas, r):
    print(f'Boundary Index: {boundary_index}')
    # Assuming boundaries are 1-indexed
    if boundary_index == 1:
        thicknesses = np.repeat(thicknesses, len(gammas[0]), axis=0).reshape(-1, len(gammas[0]))
        return r[0] * np.exp(-2 * gammas[0] * thicknesses[0])
    else:
        r_prev = recursive_ref_coeff(boundary_index - 1, thicknesses, gammas, r)
        # print(r_prev)
        # After resolving recursively:

        # TODO: understand thickness matrix shapes
        # thicknesses1 = np.repeat(thicknesses, len(gammas[0]), axis=0).reshape(-1, len(gammas[0]))
        # print(f'Thicknesses shape: {thicknesses1}')

        thicknesses = np.array(thicknesses)[:, np.newaxis]
        # print(f'Thicknesses shape: {thicknesses}')
        total_phase = np.sum(gammas[:boundary_index-1] * thicknesses[:boundary_index-1], axis=0)
        r_total = r_prev + (1 - r_prev ** 2) * r[boundary_index-1] * np.exp(-2 * total_phase)
        # TODO: at some point, maybe make sure that you don't need to have infinite reflections... last tested, only a few tenths of dB difference so....
        return r_total

# TODO: turn this into a recursive function so it can handle any number of layers atumoatically
def compute_g_truncated(r, gamma, thicknesses):
    d = np.array(thicknesses)
    g1 = r[0] * np.exp(-2 * gamma[0] * d[0])

    g2 = r[1] * (1 - r[0]**2) * np.exp(-2 * (gamma[0]*d[0] + gamma[1]*d[1]))

    g3 = r[2] * (1 - r[0]**2) * (1 - r[1]**2) * np.exp(-2 * (gamma[0]*d[0] + gamma[1]*d[1] + gamma[2]*d[2]))

    g = g1 + g2 + g3
    return g

# Input list of materials and thicknesses, frequency range
# Output S11 over frequency range
def s11_from_1D_layers(materials, thicknesses, f_start, f_stop, num_points):
    if (len(materials) < 2) or (len(thicknesses) < 2):
        print('Need at least two materials and two thicknesses to compute S11.')
        return None
    f = np.linspace(f_start, f_stop, num_points)
    dielectric_model_data = {}
    gammas = {}
    impedances = {}
    for material in materials:
        tissue_data = get_tissue_data_raw(material)
        dielectric_model_data[material] = dielectric_model_4_cole_cole(tissue_data, f)
        gammas[material] = calc_gamma(dielectric_model_data[material], f)
        impedances[material] = calc_impedance(dielectric_model_data[material], f)
    
    # Compute S11 using transmission line theory
    gamma = np.array([np.array(gammas[m]) for m in materials])
    Z = np.array([np.array(impedances[m]) for m in materials])
    r = (Z[1:] - Z[:-1]) / (Z[1:] + Z[:-1])

    # s11 = recursive_ref_coeff(len(materials) - 1, thicknesses, gamma, r)
    s11 = compute_g_truncated(r, gamma, thicknesses)

    return s11

def read_s11_file(filepath):
    s11_raw = pd.read_csv(filepath, header=1).to_numpy()
    if 'xf' in filepath.lower():
        s11_raw[:,1] = 20 * np.log10(np.abs(s11_raw[:,1] + 1j * s11_raw[:,2]))
    return s11_raw

# FDTD specific helper functions
def get_debye_parameters(tissue):
    tissue_data = get_tissue_data_raw(tissue)
    debye_params = {
        'e_inf': tissue_data[1],
        'e_d' : tissue_data[2],
        'tau' : tissue_data[3],
        'cond': tissue_data[14]
    }
    return debye_params

# Test function wrappers to run some general functionality verifications
def test_tissue_model(tissue):
    a = get_tissue_data_raw(tissue)
    f = np.linspace(1e9, 10e9, 9)
    muscle = dielectric_model_4_cole_cole(a, f)
    print(muscle['er'])
    print(muscle['cond'])

    plt.plot(f, muscle['er'])
    plt.grid()
    plt.show()

def test_s11_from_1D_layers(materials, thicknesses):
    f_start = 1e9
    f_stop = 18e9
    num_points = 1701
    s11 = s11_from_1D_layers(materials, thicknesses, f_start, f_stop, num_points)
    plt.plot(np.linspace(f_start,f_stop,num_points), 20 * np.log10(np.abs(s11)))
    plt.ylim(-60, 0)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('S11 (dB)')
    plt.grid()
    plt.show()

def compare_s11_to_files(materials, thicknesses, filepaths, fdtd_timesteps=20000, fdtd_source_location=0.01, include_external_files=False):
    # FDTD Section
    # Default max timesteps = 5000
    # Default source location is 10mm
    import fdtd
    fdtd.set_e_field_monitor(fdtd_source_location, fdtd_timesteps)
    e_source, e_reflected = fdtd.run_fdtd_simulation(materials, thicknesses, fdtd_source_location, fdtd_timesteps)
    f_fdtd, s11_fdtd_dB =  fdtd.S11_dB(e_source, e_reflected)
    plt.plot(f_fdtd, s11_fdtd_dB, label='1D FDTD S11')

    # Theoretical Section
    f_start = 1e9
    f_stop = 18e9
    num_points = 1701
    frequencies = np.linspace(f_start, f_stop, num_points)
    s11 = s11_from_1D_layers(materials, thicknesses, f_start, f_stop, num_points)
    plt.plot(frequencies, 20 * np.log10(np.abs(s11)), label='Theoretical S11')
    if include_external_files:
        for filepath in filepaths:
            s11_file = read_s11_file(filepath)
            plt.plot(s11_file[:,0]*1e9, s11_file[:,1], label=f'S11 from {filepath.split("_")[-1].replace(".csv","").upper()}')
    plt.xlim(1e9, 18e9)
    plt.ylim(-60, 0)
    plt.xticks(np.arange(1e9, 19e9, 1e9), labels=[str(int(x/1e9)) for x in np.arange(1e9, 19e9, 1e9)])
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('S11 (dB)')
    plt.legend()
    plt.grid()
    plt.show()
    
if __name__ == "__main__":
    thicknesses = [0.10, 0.01, 0.003, 0.005] # in meters
    materials = ['Air1', 'Skin', 'Fat', 'Muscle']
    base_filepath = 'C:\\Users\\corpu\\OneDrive - University of Kansas\\Applied EM Lab Work\\I2S Remote Work\\sim_eval_s11\\'
    filepaths = ['1-1-4-1_hfss.csv',
                #  '1-1-4-1_xf.csv',
                # '1-1-4-1_100mmx100mm_xf.csv',
                # '1-1-4-1_100mmx100mm_ABSxy_xf.csv',
                '1-1-4-1_500mmx500mm_ABSxy_xf.csv',
                # '1-1-4-1_2mmx150mm_xf.csv',
                # '1-1-4-1_10mmx150mm_xf.csv',
                # '1-1-4-1_10mmx150mm_20x160mmwg_xf.csv',
                # '1-1-4-1_10mmx150mm_20x30mmwg_swapbc_xf.csv',
                # '1-1-4-1_10mmx150mm_150x20mmwg_swapbc_xf.csv',
                # '1-1-4-1_150mmx150mm_xf.csv',
                 ]
    filepaths = [base_filepath + fp for fp in filepaths]
    # test_tissue_model('Blood')
    # test_s11_from_1D_layers(materials, thicknesses)
    compare_s11_to_files(materials, thicknesses, filepaths)
