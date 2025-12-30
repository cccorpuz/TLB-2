# 1D FDTD Function implementations
# Primary workflow
# 1. Establish number of layers
# 2. Establish all materials to be used
# 3. Assign materials and thicknesses to layers
# 4. Define excitation location and waveform
# 5. Run FDTD simulation
# 6. Process results (S11, SAR, etc.)

import csv
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import os
import pandas as pd

from datetime import datetime

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
dx = 0.0005  # initial spatial step size (m)\
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

source_location = 0.0001 # in meters, to be normalized later
normalized_layer_sizes = np.array([])

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
    global source_location, normalized_layer_sizes

    # Grid settings
    num_layers = len(thicknesses) - 1
    ppw = ppw_setting

    dx = (c / f_max) / ppw # tenth of a wavelength (m)
    actual_dt = 1 / f_max # time step size (s)
    const_dt = Sc * dx / c # Courant condition time step size (s)

    normalized_layer_sizes = np.array(thicknesses) / dx
    source_location = int(source_loc / dx)
    grid_size = int(np.sum(normalized_layer_sizes))
    print(normalized_layer_sizes)

    # Initialize fields and coefficients
    e_field = np.zeros(grid_size)
    e_field_prev = np.zeros(grid_size)
    h_field = np.zeros(grid_size)
    j_pol = np.zeros(grid_size)

    Cje = np.zeros(grid_size)
    Cjj = np.zeros(grid_size)
    Chyh = np.zeros(grid_size)
    Chye = np.zeros(grid_size)
    Cezhj = np.ones(grid_size)
    Ceze = np.ones(grid_size)

    # Material property integration into FDTD grid
    er = np.ones(grid_size)
    ed = np.ones(grid_size)
    tau = np.ones(grid_size)
    sigma = np.zeros(grid_size)

    for material in materials:
        print(material)
        tissue_data = get_debye_parameters(material)
        e_infs = np.concatenate((e_infs,[tissue_data['e_inf']]))
        e_ds = np.concatenate((e_ds, [tissue_data['e_d']]))
        taus = np.concatenate((taus, [tissue_data['tau']]))
        conductivities = np.concatenate((conductivities, [tissue_data['cond']]))

    for i in range(num_layers):
        layer_start = int(np.sum(normalized_layer_sizes[:i+1])) # subtract one because of zero indexing
        layer_end = int(np.sum(normalized_layer_sizes[:i+2]))
        for j in range(layer_start, layer_end):
            er[j] = e_infs[i]
            ed[j] = e_ds[i]
            tau[j] = taus[i]
            sigma[j] = conductivities[i]

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
        # Chye[i] = (const_dt / (u0 * ur * dx)) / (1 + (s_m * const_dt) / (2 * u0 * ur))
        Chye[i] = np.sqrt(e0 / u0)

    # Electric field
    for i in range(grid_size):
        # Cezhj[i] = (imp0 * Sc / er[i]) / (1 + (sigma[i] * const_dt) / (2 * e0 * er[i]) + (Cje[i] * imp0 * Sc) / (2 * er[i]))
        # Ceze[i] = (1 - (sigma[i] * const_dt) / (2 * e0 * er[i]) + (Cje[i] * imp0 * Sc) / (2 * er[i])) / (1 + (sigma[i] * const_dt) / (2 * e0 * er[i]) + (Cje[i] * imp0 * Sc) / (2 * er[i]))
        
        # See Chapter 5.7
        # TODO: find a place to calculate alpha based on frequency and medium
        # TODO: use alpha to calculate skin depth and translate to number of spatial steps N_L * dx
        # TODO: use NL to complete the "st" term below that is used so that we don't need to rely on const_dt?
        # st = np.pi / ppw * Sc * np.sqrt((1 + (ppw ** 2) / (2 * np.pi * np.pi * ))**2 - 1)
        Ceze[i] = (1 - sigma[i] * const_dt / (2 * er[i] * e0)) / (1 + sigma[i] * const_dt / (2 * er[i] * e0))
        Cezhj[i] = (const_dt / (er[i] * e0 * dx)) / (1 + sigma[i] * const_dt / (2 * er[i] * e0))

    # Print out parameters
    print(f"FDTD Simulation Initialized:\n")
    print(f"  Total Grid Size: {grid_size} cells")
    print(f"  Spatial Step Size (dx): {dx} m")
    print(f"  Time Step Size (dt): {const_dt} s")
    print(f"  Points per Wavelength (ppw): {ppw}")
    print(f"  Source Location: {source_location} cells ({source_location * dx} m)")
        

################################################################################
# FDTD Source Setup
def gaussian_pulse(qtime, location, delay): 
    width = ppw # I think that ppw translates to time because time and space are tied together via the Courant condition and max frequency of interest here
    arg = ((qtime - location - delay) / width) ** 2
    return np.exp(-arg)

################################################################################
# FDTD Update Equations
def update_magnetic_fields():
    global h_field
    for i in range(0,grid_size-1): # discretized around half time steps
        h_field[i] = Chyh[i] * h_field[i] + Chye[i] * (e_field[i + 1] - e_field[i])
    return

def update_polarization_currents():
    global j_pol, e_field, e_field_prev
    for i in range(grid_size - 1): # discretized around 1/2 time steps
        j_pol[i] = 1 / (2 * dx) * ((1 + Cjj[i]) * dx * j_pol[i] + Cje[i] * (e_field[i] - e_field_prev[i]))
    return

def update_electric_fields():
    global e_field, e_field_prev
    e_field_prev = np.copy(e_field)
    for i in range(1,grid_size): # discretized around full time steps
        e_field[i] = Ceze[i] * e_field[i] + Cezhj[i] * ((h_field[i] - h_field[i-1]) - (1 / 2 * (1 + Cjj[i]) * dx * j_pol[i]))
        # e_field[i] = Ceze[i] * e_field[i] + Cezhj[i] * ((h_field[i] - h_field[i-1]))# - (1 / 2 * (1 + Cjj[i]) * dx * j_pol[i]))
    
    # Update the polarization currents in this step too, since we ultimately only care about the electric and magnetic fields
    update_polarization_currents()

    return

def tfsf_update(field):
    # NOTE and TODO: this function only assumes a Gaussian pulse source for right now
    global e_field, h_field, excitation_complete, q_timestep, stop_condition_reached, source_location
    if field == 'e':
        e_field[source_location] += gaussian_pulse(q_timestep + 0.5, source_location - 0.5, 0)
    elif field == 'h':
        h_field[source_location-1] -= np.sqrt(er[source_location-1] * e0 / u0) * gaussian_pulse(q_timestep, source_location, 0) 

    return

def abc_update(field):
    global e_field, h_field
    if field == 'e':
        e_field[0] = e_field[1]
    elif field == 'h':
        h_field[grid_size - 1] = h_field[grid_size - 2]

    return

################################################################################
# FDTD Snapshot
def e_field_snapshot(index=None):
    # global e_field_snapshots
    # e_field_snapshots = np.concatenate((e_field_snapshots, np.copy(e_field)))

    with open(f'e_field_log{index}.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(e_field)
    return

def save_e_field_snapshot(index=None):
    global e_field_snapshots
    # Increae index if needed
    if index is None:
        log_index = 0
        while True:
            filename = f'e_field_log_sim{log_index}.csv'
            if not os.path.exists(filename):
                break
            log_index += 1
    else:
        filename = f'e_field_log_sim{index}.csv'
    
    # Reshape and save
    e_field_snapshots = np.reshape(e_field_snapshots, (grid_size, -1)).T
    np.savetxt(filename, e_field_snapshots, delimiter=",")
    print(f'E-field snapshots saved to {filename}.')
    return

################################################################################
# FDTD Energy Tracking
def update_total_energy():
    global total_energy, stop_condition_reached
    for i in range(grid_size):
        total_energy += 0.5 * e0 * er[i] * (e_field[i] ** 2)
    return

def check_total_energy():
    global stop_condition_reached
    print(f'Total energy at time step {q_timestep}:\t {10 * np.log10(total_energy)} dB')
    if (total_energy < threshold_energy) and excitation_complete:
        stop_condition_reached = True
    return

################################################################################
def run_fdtd_simulation(materials, layer_thicknesses, f_max, source_loc, max_time_steps, threshold=1e-3):
    e_field_log_index_name = 'test'
    init_scenario(layer_thicknesses, materials, f_max, source_loc, ppw_setting=10)
    
    # delete existing file if exists
    if os.path.exists(f'e_field_log{e_field_log_index_name}.csv'):
        os.remove(f'e_field_log{e_field_log_index_name}.csv')
        print(f'Existing file {f'e_field_log{e_field_log_index_name}.csv'} found and deleted.')
    
    global q_timestep, total_energy
    end_time = datetime.now()
    last_print_time = datetime.now()
    total_energy = 0.0
    for t in range(max_time_steps):
        q_timestep = t
        # total_energy = 0.0
        # print(f'Timestep: {q_timestep}, Total_energy: {total_energy}, E-Field at Source: {e_field[source_location]}')
        abc_update('h')
        update_magnetic_fields()
        tfsf_update('h')
        
        abc_update('e')
        update_electric_fields()
        tfsf_update('e')

        e_field_snapshot(e_field_log_index_name)
        # update_total_energy()

        # # timing to check for when to print energy!
        # end_time = datetime.now()
        # timediff = end_time - last_print_time
        # if int((timediff.total_seconds() + 1) % 3) == 0:
        #     check_total_energy()
        #     print(timediff.total_seconds())
        #     last_print_time = datetime.now()

        # if stop_condition_reached:
        #     print(f'Stop condition reached at time step {q_timestep}.')
        #     break
    print("FDTD Simulation Complete.")
    # save_e_field_snapshot(e_field_log_index_name)

################################################################################
if __name__ == "__main__":
    # Example usage
    materials = ['Air1', 'Air4', 'Air1']
    layer_thicknesses = [0.11, 0.003, 0.005]  # in meters
    f_max = 100e9  # 10 GHz
    source_loc = 0.01  # 0 mm from the left boundary
    max_time_steps = 1000

    run_fdtd_simulation(materials, layer_thicknesses, f_max, source_loc, max_time_steps)

    xs = np.linspace(0,grid_size-1,grid_size) # column, grid location
    ts = np.linspace(0,q_timestep-1,q_timestep).astype(int) # row, time step

    e_field_snapshots = np.loadtxt(f'e_field_logtest.csv', delimiter=',')
    fig, ax = plt.subplots()
    ax.set(xlim=[0, np.size(e_field_snapshots[0])], ylim=[-1, 2], xlabel='Grid Location', ylabel='E-Field Intensity')
    ax.grid()

    boundary_positions = []
    for i in range(1, len(normalized_layer_sizes)):  # Skip first (always 0)
        boundary_pos = int(np.sum(normalized_layer_sizes[:i+1]))
        boundary_positions.append(boundary_pos)
        ax.axvline(x=boundary_pos, color='r', linestyle='--', alpha=0.5, 
                label=f'Boundary {i}' if i == 1 else '')
    line, = ax.plot([], [], "r-")
    def update(t):
        line.set_data(xs, e_field_snapshots[t])
    
    ani = FuncAnimation(fig, update, frames=ts, interval=10, repeat=True)
    plt.show()
