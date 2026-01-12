"""
NaCl Phonon Calculation: MACE vs MACE+PME
=========================================

Compare phonon dispersion of NaCl with and without long-range
electrostatics to observe LO-TO splitting.

Expected: MACE alone will miss LO-TO splitting at Γ point,
while MACE+PME should recover it.

Experimental values for NaCl at Γ:
- TO: ~5 THz (167 cm⁻¹)
- LO: ~8 THz (267 cm⁻¹)
- Splitting: ~3 THz (100 cm⁻¹)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from ase.build import bulk
from ase.phonons import Phonons
from ase.dft.kpoints import bandpath

# Unit conversion: eV to THz
# ω (THz) = E (eV) / ℏ, where ℏ = 4.136e-15 eV·s
# So: ω (THz) = E (eV) × 241.799 THz/eV
EV_TO_THZ = 241.799

# Configuration
SUPERCELL = (3, 3, 3)  # 3x3x3 supercell for phonons
DELTA = 0.01  # Finite displacement in Angstrom
NACL_LATTICE = 5.64  # Experimental lattice constant in Angstrom

# NaCl charges
NACL_CHARGES = {'Na': 1.0, 'Cl': -1.0}


def create_nacl():
    """Create NaCl primitive cell"""
    nacl = bulk('NaCl', 'rocksalt', a=NACL_LATTICE)
    return nacl


def calculate_phonons(atoms, calc, name, supercell=SUPERCELL, delta=DELTA):
    """
    Calculate phonon band structure using finite displacements.
    
    Parameters
    ----------
    atoms : ase.Atoms
        Primitive cell
    calc : Calculator
        ASE calculator
    name : str
        Name for saving files
    supercell : tuple
        Supercell size for force constants
    delta : float
        Displacement magnitude in Angstrom
        
    Returns
    -------
    band_structure : dict
        Contains 'path', 'frequencies', 'special_points'
    """
    
    workdir = Path(f'phonon_{name}')
    workdir.mkdir(exist_ok=True)
    
    # Set calculator
    atoms.calc = calc
    
    # Initialize phonon calculation
    ph = Phonons(atoms, calc, supercell=supercell, delta=delta, name=str(workdir / name))
    
    # Run force calculations (this takes time)
    print(f"Running phonon calculations for {name}...")
    print(f"  Supercell: {supercell}")
    print(f"  Displacement: {delta} Å")
    
    ph.run()
    
    # Read forces and assemble dynamical matrix
    ph.read(acoustic=True)
    
    # Define path in BZ: Γ-X-K-Γ-L
    path = atoms.cell.bandpath('GXKGL', npoints=100)
    
    # Get band structure
    bs = ph.get_band_structure(path)
    
    # Store frequencies directly (bs object might change after clean)
    frequencies_eV = np.array(bs.energies).copy()
    
    # Debug: check actual data format
    print(f"[DEBUG] Band structure info for {name}:")
    print(f"  frequencies_eV.shape = {frequencies_eV.shape}")
    print(f"  path.kpts.shape = {path.kpts.shape}")
    
    # Clean up
    ph.clean()
    
    return {
        'path': path,
        'band_structure': bs,
        'frequencies_eV': frequencies_eV,  # Use stored copy
        'special_points': path.special_points,
    }


def plot_comparison(results_mace, results_pme, output_file='nacl_phonon_comparison.png'):
    """
    Plot phonon bands comparing MACE vs MACE+PME
    """
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    
    titles = ['MACE only', 'MACE + PME (Long-range Coulomb)']
    results_list = [results_mace, results_pme]
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    
    for ax, results, title in zip(axes, results_list, titles):
        path = results['path']
        
        # Use stored frequencies (not bs.energies which might be stale)
        freqs_eV = results['frequencies_eV']
        print(f"[DEBUG] {title}: freqs_eV.shape = {freqs_eV.shape}, ndim = {freqs_eV.ndim}")
        
        # ASE BandStructure might have shape:
        # (n_spins, n_kpts, n_bands) - electronic
        # (n_kpts, n_bands) - phonons
        # Handle both cases
        if freqs_eV.ndim == 3:
            # Has spin dimension, take first
            freqs_eV = freqs_eV[0]
        
        freqs_THz = freqs_eV * EV_TO_THZ
        
        # Ensure 2D
        if freqs_THz.ndim == 1:
            freqs_THz = freqs_THz.reshape(1, -1)
        
        n_kpts, n_bands = freqs_THz.shape
        print(f"[DEBUG] After reshape: n_kpts={n_kpts}, n_bands={n_bands}")
        
        # Get x-axis
        if n_kpts > 1:
            x = np.linspace(0, 1, n_kpts)
        else:
            x = np.array([0.0])
        
        # Plot each band
        for i in range(n_bands):
            if n_kpts > 1:
                ax.plot(x, freqs_THz[:, i], '-', color=colors[i % len(colors)], 
                        alpha=0.8, linewidth=1.5)
            else:
                # Single point - plot as horizontal line or marker
                ax.axhline(y=freqs_THz[0, i], color=colors[i % len(colors)],
                          alpha=0.8, linewidth=1.5, linestyle='--')
        
        ax.set_title(title, fontsize=14)
        ax.set_ylabel('Frequency (THz)' if ax == axes[0] else '')
        ax.set_xlabel('Wave vector')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 10)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        # Reference lines
        ax.axhline(y=5.0, color='green', linestyle=':', alpha=0.7, 
                   label='TO exp (~5 THz)')
        ax.axhline(y=8.0, color='red', linestyle=':', alpha=0.7, 
                   label='LO exp (~8 THz)')
        
        # Special points
        if n_kpts > 1:
            special_points = path.special_points
            kpts = path.kpts
            special_x = []
            special_labels = []
            
            for name, kpt in special_points.items():
                distances = np.linalg.norm(kpts - kpt, axis=1)
                idx = np.argmin(distances)
                special_x.append(x[idx])
                label = 'Γ' if name.lower() in ['g', 'gamma'] else name
                special_labels.append(label)
            
            sorted_pairs = sorted(zip(special_x, special_labels))
            special_x = [p[0] for p in sorted_pairs]
            special_labels = [p[1] for p in sorted_pairs]
            
            ax.set_xticks(special_x)
            ax.set_xticklabels(special_labels)
            
            for sx in special_x:
                ax.axvline(x=sx, color='gray', linestyle='-', alpha=0.3)
        else:
            ax.set_xticks([0, 0.5, 1])
            ax.set_xticklabels(['Γ', '', 'X'])
        
        if ax == axes[1]:
            ax.legend(loc='upper right')
    
    fig.suptitle(
        'NaCl Phonon Dispersion: Effect of Long-range Electrostatics on LO-TO Splitting',
        fontsize=14, fontweight='bold'
    )
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_file}")
    return output_file


def analyze_gamma_splitting(results, name):
    """
    Analyze frequencies at Γ point to quantify LO-TO splitting.
    """
    
    # Use stored frequencies
    freqs_eV = results['frequencies_eV']
    
    print(f"\n[DEBUG] analyze {name}: freqs_eV.shape = {freqs_eV.shape}")
    
    # Handle different shapes
    if freqs_eV.ndim == 3:
        freqs_eV = freqs_eV[0]  # Remove spin dimension
    
    # Get frequencies at Γ (first point)
    gamma_idx = 0
    freqs_gamma_eV = freqs_eV[gamma_idx]
    
    # Convert eV to THz
    freqs_gamma_THz = freqs_gamma_eV * EV_TO_THZ
    
    # Sort frequencies
    freqs_sorted = np.sort(freqs_gamma_THz)
    
    # For NaCl (2 atoms), we have 6 modes at Gamma:
    # 3 acoustic (should be ~0)
    # 3 optical (TO + LO)
    
    # Filter out acoustic (near zero) - use 1 THz threshold
    optical_freqs = freqs_sorted[freqs_sorted > 1.0]
    
    print(f"\n{name} - Frequencies at Γ point:")
    print(f"  All modes (THz): {freqs_sorted}")
    print(f"  Optical modes (THz): {optical_freqs}")
    
    if len(optical_freqs) >= 2:
        # For NaCl: should have 3 optical modes (1 LO, 2 TO degenerate)
        # Estimate splitting (max - min of optical)
        to_freq = optical_freqs[0]  # Lowest optical = TO
        lo_freq = optical_freqs[-1]  # Highest optical = LO
        splitting = lo_freq - to_freq
        
        print(f"  TO frequency: {to_freq:.2f} THz (exp: ~5 THz)")
        print(f"  LO frequency: {lo_freq:.2f} THz (exp: ~8 THz)")
        print(f"  LO-TO splitting: {splitting:.2f} THz (exp: ~3 THz)")
    
    return freqs_sorted


def main():
    """Main workflow"""
    
    print("=" * 60)
    print("NaCl Phonon Calculation: MACE vs MACE+PME")
    print("=" * 60)
    
    # Create structure
    nacl = create_nacl()
    print(f"\nNaCl structure:")
    print(f"  Lattice constant: {NACL_LATTICE} Å")
    print(f"  Atoms: {nacl.get_chemical_symbols()}")
    print(f"  Cell:\n{nacl.cell.array}")
    
    # Try to import calculators
    try:
        from mace_pme_calculator import MACEPMECalculator, MACEOnlyCalculator
        import torch
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"\nUsing device: {device}")
        
        # Initialize calculators
        print("\nInitializing calculators...")
        
        # MACE only
        calc_mace = MACEOnlyCalculator(
            mace_model='mace-omat-0-medium',
            device=device
        )
        
        # MACE + PME
        calc_pme = MACEPMECalculator(
            mace_model='mace-omat-0-medium',
            charges=NACL_CHARGES,
            pme_cutoff=12.0,
            device=device
        )
        
    except ImportError as e:
        print(f"\nError importing calculators: {e}")
        print("Creating mock results for demonstration...")
        
        # Create mock results for visualization
        return create_mock_comparison()
    
    # Calculate phonons
    print("\n" + "=" * 40)
    print("Calculating MACE-only phonons...")
    print("=" * 40)
    results_mace = calculate_phonons(nacl.copy(), calc_mace, 'mace_only')
    
    print("\n" + "=" * 40)
    print("Calculating MACE+PME phonons...")
    print("=" * 40)
    results_pme = calculate_phonons(nacl.copy(), calc_pme, 'mace_pme')
    
    # Analyze Γ point
    print("\n" + "=" * 40)
    print("Analysis at Γ point")
    print("=" * 40)
    
    analyze_gamma_splitting(results_mace, "MACE only")
    analyze_gamma_splitting(results_pme, "MACE + PME")
    
    # Plot comparison
    print("\n" + "=" * 40)
    print("Generating comparison plot...")
    print("=" * 40)
    
    output_file = plot_comparison(results_mace, results_pme)
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
    
    return output_file


def create_mock_comparison():
    """
    Create mock phonon data for visualization when dependencies unavailable.
    Shows expected behavior of LO-TO splitting.
    """
    import matplotlib.pyplot as plt
    
    print("\nGenerating mock comparison (dependencies not available)...")
    
    # Create k-point path
    npts = 100
    
    # Mock frequencies for MACE only (degenerate at Gamma)
    # Acoustic branches
    acoustic = np.zeros((npts, 3))
    acoustic[:, 0] = np.linspace(0, 4, npts)  # LA
    acoustic[:, 1] = np.linspace(0, 3, npts)  # TA1
    acoustic[:, 2] = np.linspace(0, 3, npts)  # TA2
    
    # Optical branches - WITHOUT splitting (MACE only)
    optical_mace = np.zeros((npts, 3))
    optical_mace[:, 0] = 5.5 + 0.5 * np.sin(np.linspace(0, np.pi, npts))  # TO1
    optical_mace[:, 1] = 5.5 + 0.3 * np.sin(np.linspace(0, np.pi, npts))  # TO2
    optical_mace[:, 2] = 5.8 + 0.5 * np.sin(np.linspace(0, np.pi, npts))  # LO (no split!)
    
    # Optical branches - WITH splitting (MACE + PME)
    optical_pme = np.zeros((npts, 3))
    optical_pme[:, 0] = 5.0 + 0.5 * np.sin(np.linspace(0, np.pi, npts))  # TO1
    optical_pme[:, 1] = 5.0 + 0.3 * np.sin(np.linspace(0, np.pi, npts))  # TO2
    # LO with proper splitting at Gamma
    x = np.linspace(0, 1, npts)
    lo_splitting = 3.0 * np.exp(-10 * x)  # Splitting decays away from Gamma
    optical_pme[:, 2] = 5.0 + lo_splitting + 0.5 * np.sin(np.linspace(0, np.pi, npts))  # LO
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    
    x_axis = np.linspace(0, 1, npts)
    
    # MACE only
    ax = axes[0]
    for i in range(3):
        ax.plot(x_axis, acoustic[:, i], 'b-', alpha=0.7)
        ax.plot(x_axis, optical_mace[:, i], 'r-', alpha=0.7)
    
    ax.set_title('MACE only\n(No LO-TO splitting)', fontsize=14)
    ax.set_ylabel('Frequency (THz)', fontsize=12)
    ax.set_xlabel('Wave vector', fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 10)
    ax.axhline(y=5.0, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(y=8.0, color='gray', linestyle=':', alpha=0.5)
    
    # Add special points
    ax.set_xticks([0, 0.33, 0.66, 1.0])
    ax.set_xticklabels(['Γ', 'X', 'K', 'Γ'])
    
    # Annotate
    ax.annotate('TO ≈ LO\n(degenerate)', xy=(0.02, 5.6), fontsize=10, color='red')
    
    # MACE + PME
    ax = axes[1]
    for i in range(3):
        ax.plot(x_axis, acoustic[:, i], 'b-', alpha=0.7, label='Acoustic' if i == 0 else '')
    for i in range(2):
        ax.plot(x_axis, optical_pme[:, i], 'g-', alpha=0.7, label='TO' if i == 0 else '')
    ax.plot(x_axis, optical_pme[:, 2], 'r-', alpha=0.7, label='LO')
    
    ax.set_title('MACE + PME\n(LO-TO splitting recovered)', fontsize=14)
    ax.set_xlabel('Wave vector', fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 10)
    
    # Reference lines
    ax.axhline(y=5.0, color='green', linestyle=':', alpha=0.5, label='TO exp (~5 THz)')
    ax.axhline(y=8.0, color='red', linestyle=':', alpha=0.5, label='LO exp (~8 THz)')
    
    ax.set_xticks([0, 0.33, 0.66, 1.0])
    ax.set_xticklabels(['Γ', 'X', 'K', 'Γ'])
    
    # Annotate splitting
    ax.annotate('', xy=(0.02, 8.0), xytext=(0.02, 5.0),
                arrowprops=dict(arrowstyle='<->', color='purple', lw=2))
    ax.annotate('LO-TO\nsplit\n~3 THz', xy=(0.06, 6.5), fontsize=10, color='purple')
    
    ax.legend(loc='upper right', fontsize=9)
    
    fig.suptitle(
        'NaCl Phonon Dispersion: Effect of Long-range Electrostatics on LO-TO Splitting\n'
        '(Mock data for illustration)',
        fontsize=14, fontweight='bold'
    )
    
    plt.tight_layout()
    output_file = '/home/claude/mace_pme_phonon/nacl_phonon_mock.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved mock comparison: {output_file}")
    return output_file


if __name__ == "__main__":
    output = main()
