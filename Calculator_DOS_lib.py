
import numpy as np
import scipy.constants as const
from numba import jit
import pandas as pd
def dos_spin(spin):
    return 2*spin +1

def energy (mass, k, dimension = 1):
    if dimension not in [1, 2, 3]:
        raise ValueError("Dimension must be 1, 2, or 3.")
    k_arr = np.atleast_2d(k)
    if k_arr.shape[0] == 1:
        k_arr = k_arr.T
    
    num_cols = k_arr.shape[1]
    k_x = k_arr[:, 0] if num_cols >= 1 else None
    k_y = k_arr[:, 1] if num_cols >= 2 else None
    k_z = k_arr[:, 2] if num_cols >= 3 else None
    
    if dimension == 1:
        fermi_energy = const.hbar**2 * k_x**2/(2*mass)
        phonon_energy = const.hbar * k_x * const.speed_of_sound
    elif dimension == 2:
        fermi_energy = const.hbar**2 *(k_x**2 +k_y**2)/(2*mass)
        phonon_energy = const.hbar * np.sqrt(k_x**2 +k_y**2) * const.speed_of_sound
    else:
        fermi_energy = const.hbar**2 * (k_x**2+k_y**2+k_x**2)/(2*mass)
        phonon_energy = const.hbar * np.sqrt(k_x**2+k_y**2+k_z**2) * const.speed_of_sound
    return fermi_energy, phonon_energy

class DOS:
    def __init__(self, particle_type, dimension):
        self.particle_type = particle_type
        self.dimension = dimension
        self.energy = energy

        if dimension not in [1, 2, 3]:
            raise ValueError("Dimension must be 1, 2, or 3.")
    
    def particle_mass(self):
        if self.particle_type == 'electron':
            mass = const.m_e
            spin = 0.5
            return mass, dos_spin(spin)
        elif self.particle_type == 'phonon':
            mass = const.m_p
            spin = 1
            return mass, dos_spin(spin)
        elif self.particle_type == 'custom':
            name = input("Enter the name of the custom particle: ")
            mass = float(input(f"Enter the mass of {name} in kg:"))
            if mass <= 0:
                raise ValueError("Mass must be a positive value.")
            if mass > 1e-15:
                print("Warning: This mass exceeds typical particle scale. Proceeding anyway, but just for fun.")
            spin = int(input(f"Enter the spin of {name} (e.g Electron= 0.5, phonons = 1, etc...): ")) 
            return mass, name, dos_spin(spin)
        else:
            raise ValueError("Unsupported particle type. Use 'electron', 'phonon' or 'custom'.")
    
    def density_of_states(self, mass, spin=None):
        if self.dimension == 1:
            en = energy(mass, self.k, self.dimension)
            if isinstance(en, tuple):
                en = en[0]  # Use Fermi energy for electrons
            if spin is not None:
                return (self.size /(np.pi * const.hbar) * np.sqrt(mass / (2* en))* spin)
            return (self.size /(np.pi * const.hbar) * np.sqrt(mass / (2* en)))
        elif self.dimension == 2:
            if spin is not None:
                return (mass * self.size**2)/(2 * np.pi * const.hbar**2) *spin
            return (mass * self.size**2)/(2 * np.pi * const.hbar**2)
        elif self.dimension == 3:
            en = energy(mass, self.k, self.dimension)
            if isinstance(en, tuple):
                en = en[0]
            if spin is not None:
                return  ((mass**(3/2)* self.size**3)/(np.sqrt(2)*np.pi**2 * const.hbar**3)* np.sqrt(en)) *spin
            return ((mass**(3/2)* self.size**3)/(np.sqrt(2)*np.pi**2 * const.hbar**3)* np.sqrt(en))

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import hbar, electron_mass as m_e, proton_mass as m_p
plt.style.use('dark_background')  # because we're not animals


def numerical_dos_free_particles(dimension, num_k_points=400_000, energy_bins=500, 
                                 mass=m_e, k_max = 1e8, energy_max = None, spin = None):  # eV
    
    """
    Numerically compute DOS for free electrons/phonons in 1D, 2D, 3D
    Returns energy array and g(E) in states per eV per unit volume/length/area (L = 1)
    """
    E_max_est = (hbar**2 * k_max**2 / (2 * m_e)) / 1.602e-19

    # Estimate max energy from max k
    if energy_max is not None:
        k_max = k_max
        E_max_est = (hbar**2 * k_max**2 / (2 * m_e)) / 1.602e-19
    
    # Sample k-space UNIFORMLY 
    if dimension == 1:
        k = np.random.uniform(-k_max, k_max, num_k_points)  # basically infinite
        k_weights = 2e8 / num_k_points  # dk per point (total length 2e8)
    
    elif dimension == 2:
        theta = np.random.uniform(0, 2*np.pi, num_k_points)
        r = 1e8 * np.sqrt(np.random.uniform(0, 1, num_k_points))  # uniform in area
        kx = r * np.cos(theta)
        ky = r * np.sin(theta)
        k = np.stack([kx, ky])
        k_weights = (np.pi * (1e8)**2) / num_k_points  # d²k per point
    
    elif dimension == 3:
        # Marsaglia method for uniform 
        u = np.random.normal(0, 1, (3, num_k_points))
        norm = np.sqrt(np.sum(u**2, axis=0))
        k = u / norm[None,:] * (1e8 * np.cbrt(np.random.uniform(0, 1, num_k_points)))
        k_weights = (4/3 * np.pi * (1e8)**3) / num_k_points  # d³k per point
    
    # Energy: E = ħ²k²/2m  (for electrons) — convert to eV
    k_squared = np.sum(k**2, axis=0) if dimension > 1 else k**2
    E = (hbar**2 * k_squared / (2 * mass)) / 1.602e-19  # Joules → eV
    
   
    hist, bin_edges = np.histogram(E, bins=energy_bins, range=(0, E_max_est), density=False)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    dE = bin_edges[1] - bin_edges[0]
    
    # Each k-point contributes:  g(E) dE = (number of states in dE) = sum (dk_volume / (2π)^d )
    # We sampled uniformly, so total DOS = hist * k_weights / (2π)^d / dE
    dos = hist * k_weights / ((2 * np.pi)**dimension) / dE
    
    if spin is not None:
        dos *= 2 
    return bin_centers, dos

def dos_1d_chain(num_k=100_000, energy_bins=500):
    k = np.linspace(-np.pi, np.pi, num_k)
    E = -2 * np.cos(k)  # tight-binding E = -2t cos(ka), t=1
    hist, edges = np.histogram(E, bins=energy_bins, range=(-4,4), density=True)
    centers = (edges[:-1] + edges[1:])/2
    dos = hist * (2*np.pi)  # because dk integration -----  multiply by Brillouin zone length
    return centers, dos

def dos_2d_square_lattice(num_k=500_000, energy_bins=600):
    kx = np.random.uniform(-np.pi, np.pi, num_k)
    ky = np.random.uniform(-np.pi, np.pi, num_k)
    E = -4 * (np.cos(kx)*np.cos(ky))  # 2D square lattice tight-binding
    hist, edges = np.histogram(E, bins=energy_bins, range=(-8,8), density=True)
    centers = (edges[:-1] + edges[1:])/2
    dos = hist * (2*np.pi)**2  # area of 2D BZ
    return centers, dos

def dos_1d_phonons(num_k=100_000, energy_bins=400):
    k = np.linspace(-np.pi, np.pi, num_k)
    omega = 2 * np.abs(np.sin(k/2))  # 1D phonon dispersion ω = 2√(K/m)|sin(ka/2)|
    hist, edges = np.histogram(omega, bins=energy_bins, range=(0,2), density=True)
    centers = (edges[:-1] + edges[1:])/2
    dos = hist * (2*np.pi)  # same logic
    return centers, dos


def dos_calculator(model ='Free Electron', dim = 1, spin = False):
    num_k = int(input("Number of k-points: "))
    if model == 'Free Electron':
        E_num, dos_num = numerical_dos_free_particles(dim, num_k_points=num_k, spin = spin )
        result = int(input("Do you wish to see the spectrum (Press 1) or the value for a given energy (Press 2)?: "))
        if result == 1:
            plt.plot(E_num, dos_num, 'c-', lw=2, label=f'{dim}D numerical')
            plt.title(f"Spectrum of {dim}D Free Electron", fontsize=16, color='white')
            plt.xlabel("Energy (eV)", fontsize=14)
            plt.ylabel("DOS g(E)", fontsize=14)
            plt.legend()
            plt.grid(alpha=0.3)
            plt.show()
        elif result == 2:
            target_energy = float(input("Enter the target energy: "))
            if target_energy in E_num:
                index = np.where(E_num == target_energy)[0][0]
                corr_dos = dos_num[index]
                print(f"{dim}D DOS for {target_energy}eV = ", corr_dos)
            else:
                corr_dos = np.interp(target_energy, E_num, dos_num)
                print(f"{dim}D DOS for {target_energy}eV= ", corr_dos)
    elif model == '1d Chain':
        E_num, dos_num = dos_1d_chain(num_k=num_k, energy_bins=500)
        result = int(input("Do you wish to see the spectrum (Press 1) or the value for a given energy (Press 2)?: "))
        if result == 1:
            plt.plot(E_num, dos_num, 'c-', lw=2, label=f'{dim}D numerical')
            plt.title(f"Spectrum of 1D chain", fontsize=16, color='white')
            plt.xlabel("Energy (eV)", fontsize=14)
            plt.ylabel("DOS g(E)", fontsize=14)
            plt.legend()
            plt.grid(alpha=0.3)
            plt.show()
        elif result == 2:
            target_energy = float(input("Enter the target energy: "))
            if target_energy in E_num:
                index = np.where(E_num == target_energy)[0][0]
                corr_dos = dos_num[index]
                print(f"Spectrum of 1D chain for {target_energy}eV = ", corr_dos)
            else:
                corr_dos = np.interp(target_energy, E_num, dos_num)
                print(f"Spectrum of 1D chain for {target_energy}eV= ", corr_dos)
    return

def compute_dos(model, dim, num_k, target_energy=None):
    if model == 'Free Electron':
        return numerical_dos_free_particles(dim, num_k_points=num_k)
    elif model == '1D Chain':
        return dos_1d_chain(num_k=num_k)
    

# Compute the DOS from the heat capacity...Not this 
def dos_from_user_data(E, bins=500, spin_degeneracy=2, return_edges=False):
    """
    Compute the density of states g(ε) [states / (Ry · unit-cell)] 
    for a free-electron gas from a list of sampled energies ε_i.
    
    Parameters
    ----------
    E : array_like
        Single-particle energies (in the same units as the desired DOS,
        typically Rydberg or Hartree).
    bins : int or array_like, optional
        Number of bins or explicit bin edges. Default = 500.
    spin_degeneracy : int, optional
        2 for spin-unpolarised electrons, 1 for spin-polarised. Default = 2.
    return_edges : bool, optional
        If True also return the bin edges.
    
    Returns
    -------
    ε_centers : ndarray
        Centre of each energy bin.
    g_of_ε   : ndarray
        Density of states g(ε) in states / (energy · unit-cell).
    bin_edges (optional)
    """
    E = np.asarray(E)
    
    # ------------------------------------------------------------
    # 1. Determine the k-space volume per sampled k-point
    # ------------------------------------------------------------
    # The sampling is assumed to be uniform in a sphere of radius k_max
    # such that the maximum energy in the sample is roughly \hbar^2k_max^2/2m.
    # The total k-space volume of the sphere is V_k = (4\pi/3) k_max³
    # With N_k points the volume per point is d^3k = V_k / N_k
    N_k = len(E)
    E_max = E.max()
    
    # k_max corresponding to E_max (free-electron dispersion)
    k_max = np.sqrt(2 * E_max)               # units where \hbar^2/2m = 1
    
    V_k = (4.0 / 3.0) * np.pi * k_max**3      # total sampled k-volume
    dk3_per_point = V_k / N_k                 # d^3k for every sampled k-point
    
    # ------------------------------------------------------------
    # 2. Histogram the energies
    # ------------------------------------------------------------
    # We use density=False → raw counts in each bin
    hist, bin_edges = np.histogram(E, bins=bins, range=(0, None))
    
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    dE = np.diff(bin_edges)                    # width of each bin (can be variable)
    
    # Avoid division by zero for empty bins at high energy
    with np.errstate(invalid='ignore'):
        dos = hist * dk3_per_point                  # states per bin
        dos /= (2.0 * np.pi)**3                     # convert d^3k → d^3r = 1 (one unit cell)
        dos *= spin_degeneracy                      # spin up + down
        dos /= dE                                   # → states / energy per unit cell
    
    # Clean up NaNs that appear in empty high-energy bins
    dos = np.nan_to_num(dos, nan=0.0)
    
    if return_edges:
        return bin_centers, dos, bin_edges
    else:
        return bin_centers, dos