# Ideal Link Budget (iLB)
# Crispin Corpuz 
# University of Kansas
# Created 11/16

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

    # Assuming boundaries are 1-indexed
    if boundary_index == 1:
        thicknesses = np.repeat(thicknesses, len(gammas[0]), axis=0).reshape(-1, len(gammas[0]))
        return r[0] * np.exp(-2 * gammas[0] * thicknesses[0])
    else:
        r_prev = recursive_ref_coeff(boundary_index - 1, thicknesses, gammas, r)
        print(r_prev)
        # After resolving recursively:
        thicknesses = np.repeat(thicknesses, len(gammas[0]), axis=0).reshape(-1, len(gammas[0]))
        total_phase = np.sum(gammas[:boundary_index-1] * thicknesses[:boundary_index-1], axis=0)
        r_total = r_prev + (1 - r_prev ** 2) * r[boundary_index-1] * np.exp(-2 * total_phase)
        # TODO: at some point, maybe make sure that you don't need to have infinite reflections... last tested, only a few tenths of dB difference so....
        return r_total


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
    r = np.round(r, 4)

    s11 = recursive_ref_coeff(len(materials) - 1, thicknesses, gamma, r)

    return s11



# Run this to test the tissue model test and compare to IT'IS frequency chart. 
def test_tissue_model(tissue):
    a = get_tissue_data_raw(tissue)
    f = np.linspace(1e9, 10e9, 9)
    muscle = dielectric_model_4_cole_cole(a, f)
    print(muscle['er'])
    print(muscle['cond'])

    plt.plot(f, muscle['er'])
    plt.grid()
    plt.show()

def test_s11_from_1D_layers(materials):
    thicknesses = [0.10, 0.01, 0.003, 0.005] # in meters
    f_start = 1e9
    f_stop = 18e9
    num_points = 1701
    s11 = s11_from_1D_layers(materials, thicknesses, f_start, f_stop, num_points)
    plt.plot(np.linspace(f_start,f_stop,num_points), 20 * np.log10(np.abs(s11)))
    plt.grid()
    plt.show()
    
if __name__ == "__main__":
    materials = ['Air1', 'Air2', 'Air4', 'Air1']
    # test_tissue_model('Blood')
    test_s11_from_1D_layers(materials)
