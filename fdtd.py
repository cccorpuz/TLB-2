# 1D FDTD Function implementations
# Primary workflow
# 1. Establish number of layers
# 2. Establish all materials to be used
# 3. Assign materials and thicknesses to layers
# 4. Define excitation location and waveform
# 5. Run FDTD simulation
# 6. Process results (S11, SAR, etc.)

import numpy as np
import os
import pandas as pd

from idealLB import get_debye_parameters

################################################################################
# global variables
c = 3e8  # speed of light in vacuum
u0 = 4e-7 * np.pi  # permeability of free space
e0 = 8.854e-12  # permittivity of free space
imp0 = np.sqrt(u0 / e0)  # impedance of free space

thicknesses = np.array([])
e_infs = np.array([])
e_ds = np.array([])
taus = np.array([])
conductivities = np.array([])

# FDTD parameters
Sc = 1.0  # Courant number
dx = 0.001  # initial spatial step size (m)\
const_dt = 0
actual_dt = 0  
ppw = 10 # points per wavelength
grid_size = 0
max_time = 100000
q_timestep = 0 # current time step

er = np.array([])
ed = np.array([])
tau = np.array([])
sigma = np.array([])

source_location = 0 # in meters, to be normalized later

e_field = np.array([])
e_field_prev = np.array([])
h_field = np.array([])
j_pol = np.array([])

Cje = np.array([])
Cjj = np.array([])
Chyh = np.array([])
Chye = np.array([])
Cezhj = np.array([])
Ceze = np.array([])

total_energy = 0.0
threshold_energy = 1e-3
e_field_snapshots = np.array([])

excitation_complete = False
stop_condition_reached = False


################################################################################
# Initialize the FDTD simulation scenario
def init_scenario(layer_thicknesses, materials, f_max, source_loc, ppw_setting=10):
    layer_thicknesses = np.concatenate((np.array([0]),layer_thicknesses))
    if len(layer_thicknesses) - 1 == len(materials):
        global thicknesses
        thicknesses = np.array(layer_thicknesses)
    else:
        raise ValueError("Thicknesses and materials lists must have the same length.")
    
    if source_loc < 0 or source_loc >= np.sum(thicknesses):
        raise ValueError("Source location must be within the total thickness of the layers.")

    global e_infs, e_ds, taus, conductivities, dx, const_dt, actual_dt, grid_size, ppw
    global Cje, Cjj, Chyh, Chye, Cezhj, Ceze, e_field, e_field_prev, h_field, j_pol
    global er, ed, tau, sigma
    global source_location

    # Grid settings
    num_layers = len(thicknesses) - 1
    ppw = ppw_setting

    dx = (c / f_max) / ppw # tenth of a wavelength (m)
    actual_dt = 1 / f_max # time step size (s)
    const_dt = Sc * dx / c # Courant condition time step size (s)

    normalized_layer_sizes = np.array(thicknesses) / dx
    source_location = int(source_loc / dx)
    grid_size = int(np.sum(normalized_layer_sizes))

    # Initialize fields and coefficients
    e_field = np.zeros(grid_size)
    e_field_prev = np.zeros(grid_size)
    h_field = np.zeros(grid_size)
    j_pol = np.zeros(grid_size)

    Cje = np.zeros(grid_size)
    Cjj = np.zeros(grid_size)
    Chyh = np.zeros(grid_size)
    Chye = np.zeros(grid_size)
    Cezhj = np.zeros(grid_size)
    Ceze = np.zeros(grid_size)

    # Material property integration into FDTD grid
    er = np.ones(grid_size)
    ed = np.ones(grid_size)
    tau = np.ones(grid_size)
    sigma = np.zeros(grid_size)

    for material in materials:
        tissue_data = get_debye_parameters(material)
        e_infs = np.concatenate((e_infs,[tissue_data['e_inf']]))
        e_ds = np.concatenate((e_ds, [tissue_data['e_d']]))
        taus = np.concatenate((taus, [tissue_data['tau']]))
        conductivities = np.concatenate((conductivities, [tissue_data['cond']]))

    for i in range(num_layers - 1):
        layer_start = int(np.sum(normalized_layer_sizes[:i])) - 1 # subtract one because of zero indexing
        layer_end = int(np.sum(normalized_layer_sizes[:i+1])) - 1
        er[layer_start:layer_end] = e_infs[i]
        ed[layer_start:layer_end] = e_ds[i]
        tau[layer_start:layer_end] = taus[i]
        sigma[layer_start:layer_end] = conductivities[i]

    # Update Coefficients
    # Polarization current
    for i in range(grid_size):
        NT = tau[i] / const_dt
        if NT == 0.0:
            Cjj[i] = 1.0
            Cje[i] = 0.0
        else:
            Cjj[i] = (1 - 1 / (2 * NT)) / (1 + 1 / (2 * NT))
            Cje[i] = 1 / (NT * (1 + 1 / (2 * NT))) * (ed[i] / (imp0 * Sc))
    
    # Magnetic field
    s_m = 0 # assuming "magnetic conductivity" is zero, as for most paramagnetic materials
    ur = 1 # assuming non-magnetic materials
    for i in range(grid_size):
        Chyh[i] = (1 - (s_m * const_dt) / (2 * u0 * ur)) / (1 + (s_m * const_dt) / (2 * u0 * ur))
        Chye[i] = (const_dt / (u0 * ur * dx)) / (1 + (s_m * const_dt) / (2 * u0 * ur))

    # Electric field
    for i in range(grid_size):
        Cezhj[i] = (imp0 * Sc / er[i]) / (1 + (sigma[i] * const_dt) / (2 * e0 * er[i]) + (Cje[i] * imp0 * Sc) / (2 * er[i]))
        Ceze[i] = (1 - (sigma[i] * const_dt) / (2 * e0 * er[i]) + (Cje[i] * imp0 * Sc) / (2 * er[i])) / (1 + (sigma[i] * const_dt) / (2 * e0 * er[i]) + (Cje[i] * imp0 * Sc) / (2 * er[i]))

################################################################################
# FDTD Source Setup
def gaussian_pulse(time, location, delay): 
    delay = 0 # TODO: doesn't matter right now
    width = ppw # I think that ppw translates to time because time and space are tied together via the Courant condition and max frequency of interest here
    arg = ((time - location - delay) / width) ** 2
    return np.exp(-arg)

################################################################################
# FDTD Update Equations
def update_magnetic_fields():
    global h_field, e_field
    for i in range(grid_size - 1): # discretized around half time steps
        h_field[i] = Chyh[i] * h_field[i] + Chye[i] * (e_field[i + 1] - e_field[i])

def update_polarization_currents():
    global j_pol, e_field, e_field_prev
    for i in range(grid_size - 1): # discretized around 1/2 time steps
        j_pol[i] = 1 / (2 * dx) * ((1 + Cjj[i]) * dx * j_pol[i] + Cje[i] * (e_field[i] - e_field_prev[i]))

def update_electric_fields():
    global e_field, e_field_prev
    e_field_prev = np.copy(e_field)
    for i in range(grid_size - 1): # discretized around full time steps
        e_field[i] = Ceze[i] * e_field[i] + Cezhj[i] * ((h_field[i+1] - h_field[i]) - (1 / 2 * (1 + Cjj[i]) * dx * j_pol[i]))
        update_total_energy(i)
    
    # Update the polarization currents in this step too, since we ultimately only care about the electric and magnetic fields
    update_polarization_currents()

def tfsf_update():
    # NOTE and TODO: this function only assumes a Gaussian pulse source for right now
    global e_field, h_field, excitation_complete
    h_field[source_location] -= gaussian_pulse(q_timestep, source_location, 0) * Chye[source_location]
    e_field[source_location + 1] += gaussian_pulse(q_timestep + 0.5, source_location - 0.5, 0)

    if (gaussian_pulse(q_timestep + 0.5, source_location - 0.5, 0) < 1e-6) and (not excitation_complete) and q_timestep > 2000:
        print(f'Excitation complete at time step {q_timestep}.')
        excitation_complete = True

def abc_update():
    global e_field
    e_field[0] = e_field[1]
    e_field[-1] = e_field[-2]

################################################################################
# FDTD Snapshot
def e_field_snapshot():
    global e_field_snapshots
    e_field_snapshots = np.concatenate((e_field_snapshots, np.copy(e_field)))
    return

def save_e_field_snapshot():
    log_index = 0
    while True:
         filename = f'e_field_log_sim{log_index}.csv'
         if not os.path.exists(filename):
              break
         log_index += 1
    np.savetxt(filename, e_field_snapshot, delimiter=",")
    return

################################################################################
# FDTD Energy Tracking
def update_total_energy(grid_location):
    global total_energy
    total_energy += 0.5 * e0 * er[grid_location] * (e_field[grid_location] ** 2)
    return

def check_total_energy():
    global stop_condition_reached, total_energy
    if q_timestep % 10 == 0:
        print(f'Total energy at time step {q_timestep}:\t {10 * np.log10(total_energy)} dB')
    if (total_energy < threshold_energy) and (q_timestep > 2000):
        stop_condition_reached = True
    total_energy = 0.0
    return

################################################################################
def run_fdtd_simulation(materials, layer_thicknesses, f_max, source_loc, max_time_steps, threshold=1e-3):
    init_scenario(layer_thicknesses, materials, f_max, source_loc)
    global q_timestep
    for t in range(max_time_steps):
        q_timestep = t
        update_magnetic_fields()
        tfsf_update()
        abc_update()
        update_electric_fields()
        e_field_snapshot()
        check_total_energy()
        if stop_condition_reached:
            print(f'Stop condition reached at time step {q_timestep}.')
            break
    save_e_field_snapshot()

################################################################################
if __name__ == "__main__":
    # Example usage
    materials = ['Air1', 'Air4', 'Air1']
    layer_thicknesses = [0.11, 0.003, 0.005]  # in meters
    f_max = 20e9  # 10 GHz
    source_loc = 0.0  # 5 mm from the left boundary
    max_time_steps = 100000
    run_fdtd_simulation(materials, layer_thicknesses, f_max, source_loc, max_time_steps)