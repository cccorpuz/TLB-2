# Written by Crispin Corpuz
# 4/1/2026
# Takes in code from XFdtd far field and plane wave data exports, parsing them to get the S11

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class XF_nearfield:
    def __init__(self, filename):
        self.time, self.ex, self.ey, self.ez, self.arr_length = self.load_xf_nearfield_e(filename)
        self.frequencies, self.ex_f, self.ey_f, self.ez_f, self.e_f = None, None, None, None, None

    # Parse XFdtd electric field for plane wave excitation
    def load_xf_nearfield_e(self, filename):
        self.pw_data = pd.read_csv(filename, encoding='cp1252')
        time = self.pw_data['Time (ns)'].values.copy()
        ex = self.pw_data['Total E x (V/m)'].values.copy()
        ey = self.pw_data['Total E y (V/m)'].values.copy()
        ez = self.pw_data['Total E z (V/m)'].values.copy()
        arr_length = len(time)
        return time, ex, ey, ez, arr_length 
    
    def nearfield_freq(self):
        # Units
        ns = 1e-9
        dt = (self.time[1] - self.time[0]) * ns
        N = len(self.time)
        
        # rfft automatically handles real-to-complex FFT and only returns positive frequencies
        self.ex_f = np.fft.rfft(self.ex) / N
        self.ey_f = np.fft.rfft(self.ey) / N
        self.ez_f = np.fft.rfft(self.ez) / N
        
        self.frequencies = np.fft.rfftfreq(N, d=dt)

        # FIXED: Absolute value taken BEFORE squaring complex numbers
        self.e_f = np.sqrt(np.abs(self.ex_f)**2 + np.abs(self.ey_f)**2 + np.abs(self.ez_f)**2)
        
        # Note: Removed plt.show() from here so the script runs cleanly without pausing
        return
    
    def diff(self, other):
        # FIXED: Uses min() to prevent a crash if arrays are slightly different lengths
        min_n = min(len(self.time), len(other.time))
        self.ex[:min_n] -= other.ex[:min_n]
        self.ey[:min_n] -= other.ey[:min_n]
        self.ez[:min_n] -= other.ez[:min_n]
        self.e_t = np.sqrt(self.ex**2 + self.ey**2 + self.ez**2)


if __name__ == "__main__":
    base = XF_nearfield("C:\\TLB-2\\data\\baseline_nearfield_finer_broadband_04122026.csv")
    nf = XF_nearfield("C:\\TLB-2\\data\\2-3-4-5_nearfield_finer_broadband_04122026.csv")

    # Subtract incident field to get scattered field
    nf.diff(base)

    if len(base.ex) < len(nf.ex):
        pad_length = len(nf.ex) - len(base.ex)
        base.ex = np.pad(base.ex, (0, pad_length), 'constant')
        base.ey = np.pad(base.ey, (0, pad_length), 'constant')
        base.ez = np.pad(base.ez, (0, pad_length), 'constant')
        # Also need to extend the time array so the FFT dt calculation holds
        base.time = nf.time 

    # Now calculate FFTs safely
    nf.nearfield_freq()
    base.nearfield_freq()

    # Calculate the ratio (safe, because lengths and bins now match perfectly)
    ratio_dB = 20 * np.log10(np.abs(nf.e_f) / np.abs(base.e_f))

    plt.plot(base.frequencies, ratio_dB)
    
    plt.xlim(6e9, 14e9) 
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Ratio of Scattered to Incident Field (dB)")
    plt.title("Ratio of Scattered to Incident Field vs Frequency")
    plt.grid(True)
    plt.show()