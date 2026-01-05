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
f_max = 0.0 

thicknesses = np.array([])
e_infs = np.array([])
e_ds = np.array([])
taus = np.array([])
conductivities = np.array([])

# FDTD parameters
Sc = 1.0  # Courant number
dx = 1  # initial spatial step size (m)\
const_dt = 0
actual_dt = 0  
ppw = 20 # points per wavelength
grid_size = 0
max_time = 100000
q_timestep = 0 # current time step

er = np.array([])
ed = np.array([])
tau = np.array([])
sigma = np.array([])

source_location = 0.0001 # in meters, to be normalized later
source_duration = 0.0
normalized_layer_sizes = np.array([])

e_field_monitor_Location = -1.0
e_field_monitor_data = np.array([])
e_field_source = np.array([])

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

# Post Processing Variables
e_field_reflected_fft = np.array([])
e_field_source_fft = np.array([])
frequencies = np.array([])


################################################################################
# Initialize the FDTD simulation scenario
def init_scenario(layer_thicknesses, materials, source_loc, fmax=30e9, ppw_setting=15):
    layer_thicknesses = np.concatenate((np.array([0]),layer_thicknesses))
    if len(layer_thicknesses) - 1 == len(materials):
        global thicknesses
        thicknesses = np.array(layer_thicknesses)
    else:
        raise ValueError("Thicknesses and materials lists must have the same length.")
    
    if source_loc < 0 or source_loc >= np.sum(thicknesses):
        raise ValueError("Source location must be within the total thickness of the layers.")

    global e_infs, e_ds, taus, conductivities, dx, const_dt, actual_dt, grid_size, ppw, f_max
    global Cje, Cjj, Chyh, Chye, Cezhj, Ceze, e_field, e_field_prev, h_field, j_pol
    global er, ed, tau, sigma
    global source_location, normalized_layer_sizes
    f_max = fmax

    for material in materials:
        print(material)
        tissue_data = get_debye_parameters(material)
        e_infs = np.concatenate((e_infs,[tissue_data['e_inf']]))
        e_ds = np.concatenate((e_ds, [tissue_data['e_d']]))
        taus = np.concatenate((taus, [tissue_data['tau']]))
        conductivities = np.concatenate((conductivities, [tissue_data['cond']]))

    # Grid settings
    num_layers = len(thicknesses) - 1
    ppw = ppw_setting

    # Determine spatial and temporal step sizes
    lambda_min = c / (f_max * np.sqrt(np.max(e_infs)))
    min_physical_thickness = np.min(thicknesses[thicknesses>0]) / 5 # divide by 5 in order to have some margin for smallest feature
    if lambda_min / ppw < min_physical_thickness:
        dx = lambda_min / ppw
    else:
        print("Minimum physical thickness is smaller than minimum wavelength / ppw")
        dx = min_physical_thickness
    # actual_dt = 1 / f_max # time step size (s)
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

    for i in range(num_layers):
        layer_start = int(np.sum(normalized_layer_sizes[:i+1])) # subtract one because of zero indexing
        layer_end = int(np.sum(normalized_layer_sizes[:i+2]))
        # Establish overall dielectric properties
        print(layer_start)
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
# FDTD Source and Monitor Setup

# Primary broadband waveform for fdtd
def ricker_wavelet(qtime, location, delay, fc=40e9):
    global source_duration, excitation_complete
    # not optimal placement but not a painful calculation at least
    source_duration = 3 / fc # estimate of source length

    # Time in actual seconds
    t = (qtime - location - delay) * const_dt

    if t > source_duration:
        excitation_complete = True
    
    # Ricker wavelet formula
    term = (np.pi * fc * t) ** 2
    return (1 - 2 * term) * np.exp(-term)

# The following not really used but can be if you need
def gaussian_pulse(qtime, location, delay, fc=20e9): 
    width = ppw # I think that ppw translates to time because time and space are tied together via the Courant condition and max frequency of interest here
    arg = ((qtime - location - delay) * const_dt / width) ** 2
    return np.exp(-arg)


def sine_pulse(qtime, location, delay, fc=10e9):
    # fc = f_max
    duration = 1 / fc * 2
    # duration = max_time
    t = (qtime - location - delay) * const_dt
    if t >= 0 and t <= duration:
        return np.sin(2 * np.pi * fc * t)
    else:
        return 0.0

def set_e_field_monitor(location, max_time_steps):
    # give e_field monitor location in meters, and this automatically converts to grid index
    global e_field_monitor_Location, e_field_monitor_data, e_field_source
    e_field_source = np.zeros(max_time_steps)
    e_field_monitor_data = np.zeros(max_time_steps)
    e_field_monitor_Location = int(location / dx)
    return

################################################################################
# Manual DFT calculations if needed
# otherwise, using numpy.fft.fft for speed
def dft(signal):
    N = len(signal)
    n = np.arange(N, dtype='complex128') # evenly spaced
    k = n.reshape((N,1)) # turn n into a column vector
    exp = np.exp(-2j * np.pi * k * n / N)
    signal_dft = np.dot(exp, signal) / N
    return signal_dft

def dft_frequencies(timesteps, duration):
    sampling_rate = (timesteps / duration) 
    freq_resolution = sampling_rate / timesteps
    frequencies = np.arange(0, sampling_rate, freq_resolution)
    return frequencies

def magnitude_spectrum(signal_dft):
    N = len(signal_dft)
    magnitudes = 2 * np.abs(signal_dft[:N//2])
    return magnitudes


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
        e_field[source_location] += ricker_wavelet(q_timestep, source_location, 0)
    elif field == 'h':
        h_field[source_location] -= np.sqrt(er[source_location] * e0 / u0) * ricker_wavelet(q_timestep, source_location, 0) 

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
def run_fdtd_simulation(materials, layer_thicknesses, source_loc, max_time_steps, f_max=30e9,threshold=1e-3):
    e_field_log_index_name = 'test'
    init_scenario(layer_thicknesses, materials, source_loc, f_max, ppw_setting=10)
    
    # delete existing file if exists
    if os.path.exists(f'e_field_log{e_field_log_index_name}.csv'):
        os.remove(f'e_field_log{e_field_log_index_name}.csv')
        print(f'Existing file {f'e_field_log{e_field_log_index_name}.csv'} found and deleted.')
    
    global q_timestep, total_energy, e_field_monitor_data, e_field_source
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

        if e_field_monitor_Location >= 0.0:
            e_field_monitor_data[t] = e_field[e_field_monitor_Location]
            if excitation_complete:
                e_field_source[t] = 0.0
            else:
                e_field_source[t] = e_field[source_location+1]

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
    return e_field_source, e_field_monitor_data


# calculate S11 in dB and associated frequencies
def S11_dB(e_field_source, e_field_monitor_data):
    global e_field_reflected_fft, e_field_source_fft, frequencies, const_dt

    if (len(e_field_monitor_data) != len(e_field_source)):
        print("S11 cannot be calculated due to differing input array lengths")

    N = len(e_field_monitor_data)    
    e_field_reflected_fft = np.fft.fft(e_field_monitor_data) / N
    e_field_source_fft = np.fft.fft(e_field_source) / N
    frequencies = np.fft.fftfreq(N, d=const_dt)[:N//2]

    R = np.abs(e_field_reflected_fft[0:N//2] / e_field_source_fft[0:N//2])
    return frequencies, 20*np.log10(R)

################################################################################
if __name__ == "__main__":
    # Setups
    materials = ['Air1', 'Skin', 'Fat', 'Muscle']
    layer_thicknesses = [0.10, 0.01, 0.003, 0.005] # in meters 

    max_time_steps = 20000
    source_loc = 0.01  # 0 mm from the left boundary
    
    set_e_field_monitor(location = 0.0, max_time_steps = max_time_steps)

    e_source, e_reflected = run_fdtd_simulation(materials, layer_thicknesses, source_loc, max_time_steps)

    ################################################################################
    # TODO: turn this post-processing into a function to call from other files
    # Post-processing data
    f, s11_dB =  S11_dB(e_source, e_reflected)
    plt.plot(f, s11_dB)

    # debug plots
    # plt.plot(f,20*np.log10(np.abs(e_field_source_dft[0:max_time_steps//2])))
    # plt.plot(f,20*np.log10(np.abs(e_field_reflected_dft[0:max_time_steps//2])))
    # plt.plot(e_field_monitor_data)
    # plt.plot(e_field_source)

    # set plot boundaries
    plt.xlim(1e9, 20e9)
    plt.ylim(-60, 0)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('S11 (dB)')
    plt.xticks(np.arange(1e9, 21e9, 1e9), labels=[str(int(x/1e9)) for x in np.arange(1e9, 21e9, 1e9)])
    plt.grid()
    plt.show()
    
    ################################################################################
    # Time Domain Animation
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
    # Save as MP4 video
    print("Saving MP4...")
    ani.save('fdtd_animation.mp4', writer='ffmpeg', fps=30, dpi=150)

    # Save as GIF
    print("Saving GIF...")
    ani.save('fdtd_animation.gif', writer='pillow', fps=30, dpi=100)

    print("Animation files saved!")
    plt.show()
