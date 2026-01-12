"""
NaCl Phonon Calculation: MACE vs MACE+PME
=========================================

Compare phonon dispersion with and without long-range electrostatics.

Experimental NaCl at Γ:
- TO: ~5 THz
- LO: ~8 THz
- Splitting: ~3 THz
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from ase.build import bulk
from ase.phonons import Phonons
from ase.optimize import BFGS

# Unit conversion: eV to THz
EV_TO_THZ = 241.799

# Configuration
SUPERCELL = (3, 3, 3)
DELTA = 0.01
NACL_LATTICE = 5.64

# NaCl charges
NACL_CHARGES = {'Na': 1.0, 'Cl': -1.0}


def get_strain_filter():
    """Get appropriate filter for cell optimization based on ASE version."""
    try:
        from ase.filters import StrainFilter
        return StrainFilter
    except ImportError:
        pass
    try:
        from ase.filters import ExpCellFilter
        return ExpCellFilter
    except ImportError:
        pass
    try:
        from ase.filters import FrechetCellFilter
        return FrechetCellFilter
    except ImportError:
        pass
    raise ImportError("No suitable cell filter found in ASE")


def create_nacl():
    """Create NaCl primitive cell"""
    return bulk('NaCl', 'rocksalt', a=NACL_LATTICE)


def optimize_structure(atoms, calc, name, fmax=0.01):
    """Optimize atomic positions and lattice parameters."""
    
    StrainFilter = get_strain_filter()
    
    print(f"\n{'='*40}")
    print(f"Optimizing structure with {name}")
    print(f"{'='*40}")
    
    atoms = atoms.copy()
    atoms.calc = calc
    
    e_init = atoms.get_potential_energy()
    cell_init = atoms.cell.lengths()
    print(f"Initial energy: {e_init:.4f} eV")
    print(f"Initial cell: {cell_init}")
    
    sf = StrainFilter(atoms)
    opt = BFGS(sf, logfile=f'opt_{name}.log')
    opt.run(fmax=fmax)
    
    e_final = atoms.get_potential_energy()
    cell_final = atoms.cell.lengths()
    print(f"Final energy: {e_final:.4f} eV")
    print(f"Final cell: {cell_final}")
    print(f"Energy change: {e_final - e_init:.4f} eV")
    
    return atoms


def check_pme_contribution(atoms, calc_pme):
    """Print detailed energy breakdown for MACE+PME calculator."""
    print(f"\n{'='*40}")
    print("PME Energy Contribution Check")
    print(f"{'='*40}")
    
    atoms = atoms.copy()
    atoms.calc = calc_pme
    
    e_total = atoms.get_potential_energy()
    
    e_mace = calc_pme.results.get('energy_mace', 0)
    e_pme = calc_pme.results.get('energy_pme', 0)
    
    print(f"Total energy:  {e_total:.6f} eV")
    print(f"MACE energy:   {e_mace:.6f} eV")
    print(f"PME energy:    {e_pme:.6f} eV")
    if abs(e_total) > 1e-10:
        print(f"PME fraction:  {abs(e_pme/e_total)*100:.2f} %")
    
    print(f"\n(Reference: NaCl Madelung energy ~ -8.9 eV/f.u.)")
    
    return e_mace, e_pme


def calculate_phonons(atoms, calc, name, supercell=SUPERCELL, delta=DELTA):
    """Calculate phonon band structure using finite displacements."""
    
    workdir = Path(f'phonon_{name}')
    workdir.mkdir(exist_ok=True)
    
    atoms.calc = calc
    
    ph = Phonons(atoms, calc, supercell=supercell, delta=delta, name=str(workdir / name))
    
    print(f"Running phonon calculations for {name}...")
    print(f"  Supercell: {supercell}")
    print(f"  Displacement: {delta} Å")
    
    ph.run()
    ph.read(acoustic=True)
    
    path = atoms.cell.bandpath('GXKGL', npoints=100)
    bs = ph.get_band_structure(path)
    
    frequencies_eV = np.array(bs.energies).copy()
    
    print(f"[DEBUG] frequencies_eV.shape = {frequencies_eV.shape}")
    print(f"[DEBUG] path.kpts.shape = {path.kpts.shape}")
    
    ph.clean()
    
    return {
        'path': path,
        'band_structure': bs,
        'frequencies_eV': frequencies_eV,
        'special_points': path.special_points,
    }


def analyze_gamma_splitting(results, name):
    """Analyze frequencies at Γ point to quantify LO-TO splitting."""
    
    freqs_eV = results['frequencies_eV']
    
    print(f"\n[DEBUG] analyze {name}: freqs_eV.shape = {freqs_eV.shape}")
    
    if freqs_eV.ndim == 3:
        freqs_eV = freqs_eV[0]
    
    gamma_idx = 0
    freqs_gamma_eV = freqs_eV[gamma_idx]
    freqs_gamma_THz = freqs_gamma_eV * EV_TO_THZ
    freqs_sorted = np.sort(freqs_gamma_THz)
    
    # Filter optical modes (> 1 THz)
    optical_freqs = freqs_sorted[freqs_sorted > 1.0]
    
    print(f"\n{name} - Frequencies at Γ point:")
    print(f"  All modes (THz): {freqs_sorted}")
    print(f"  Optical modes (THz): {optical_freqs}")
    
    if len(optical_freqs) >= 2:
        to_freq = optical_freqs[0]
        lo_freq = optical_freqs[-1]
        splitting = lo_freq - to_freq
        
        print(f"  TO frequency: {to_freq:.2f} THz (exp: ~5 THz)")
        print(f"  LO frequency: {lo_freq:.2f} THz (exp: ~8 THz)")
        print(f"  LO-TO splitting: {splitting:.2f} THz (exp: ~3 THz)")
    
    return freqs_sorted


def plot_comparison(results_mace, results_pme, output_file='nacl_phonon_comparison.png'):
    """Plot phonon bands comparing MACE vs MACE+PME"""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    
    titles = ['MACE only', 'MACE + PME (Long-range Coulomb)']
    results_list = [results_mace, results_pme]
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    
    for ax, results, title in zip(axes, results_list, titles):
        path = results['path']
        freqs_eV = results['frequencies_eV']
        
        print(f"[DEBUG] {title}: freqs_eV.shape = {freqs_eV.shape}")
        
        if freqs_eV.ndim == 3:
            freqs_eV = freqs_eV[0]
        
        freqs_THz = freqs_eV * EV_TO_THZ
        
        if freqs_THz.ndim == 1:
            freqs_THz = freqs_THz.reshape(1, -1)
        
        n_kpts, n_bands = freqs_THz.shape
        print(f"[DEBUG] After processing: n_kpts={n_kpts}, n_bands={n_bands}")
        
        x = np.linspace(0, 1, n_kpts) if n_kpts > 1 else np.array([0.0])
        
        for i in range(n_bands):
            if n_kpts > 1:
                ax.plot(x, freqs_THz[:, i], '-', color=colors[i % len(colors)], 
                        alpha=0.8, linewidth=1.5)
            else:
                ax.axhline(y=freqs_THz[0, i], color=colors[i % len(colors)],
                          alpha=0.8, linewidth=1.5, linestyle='--')
        
        ax.set_title(title, fontsize=14)
        ax.set_ylabel('Frequency (THz)' if ax == axes[0] else '')
        ax.set_xlabel('Wave vector')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 10)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        ax.axhline(y=5.0, color='green', linestyle=':', alpha=0.7, label='TO exp (~5 THz)')
        ax.axhline(y=8.0, color='red', linestyle=':', alpha=0.7, label='LO exp (~8 THz)')
        
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
    
    # Import calculators
    from mace_pme_calculator import MACEPMECalculator, MACEOnlyCalculator
    import torch
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    print("\nInitializing calculators...")
    
    calc_mace = MACEOnlyCalculator(
        mace_model='mace-omat-0-medium',
        device=device
    )
    
    calc_pme_verbose = MACEPMECalculator(
        mace_model='mace-omat-0-medium',
        charges=NACL_CHARGES,
        pme_cutoff=12.0,
        device=device,
        verbose=True
    )
    
    calc_pme = MACEPMECalculator(
        mace_model='mace-omat-0-medium',
        charges=NACL_CHARGES,
        pme_cutoff=12.0,
        device=device,
        verbose=False
    )
    
    # Step 1: Check PME contribution before optimization
    print("\n" + "=" * 60)
    print("STEP 1: Check PME contribution (before optimization)")
    print("=" * 60)
    check_pme_contribution(nacl.copy(), calc_pme_verbose)
    
    # Step 2: Optimize structures
    print("\n" + "=" * 60)
    print("STEP 2: Structure Optimization")
    print("=" * 60)
    
    nacl_opt_mace = optimize_structure(nacl.copy(), calc_mace, 'mace_only')
    nacl_opt_pme = optimize_structure(nacl.copy(), calc_pme_verbose, 'mace_pme')
    
    print(f"\n{'='*40}")
    print("Optimized lattice constants comparison:")
    print(f"  MACE only:  {nacl_opt_mace.cell.lengths()[0]:.4f} Å")
    print(f"  MACE + PME: {nacl_opt_pme.cell.lengths()[0]:.4f} Å")
    print(f"  Experiment: {NACL_LATTICE:.4f} Å")
    
    # Step 3: Check PME contribution after optimization
    print("\n" + "=" * 60)
    print("STEP 3: Check PME contribution (after optimization)")
    print("=" * 60)
    check_pme_contribution(nacl_opt_pme, calc_pme_verbose)
    
    # Step 4: Calculate phonons
    print("\n" + "=" * 60)
    print("STEP 4: Phonon Calculations")
    print("=" * 60)
    
    print("\nCalculating MACE-only phonons...")
    results_mace = calculate_phonons(nacl_opt_mace, calc_mace, 'mace_only')
    
    print("\nCalculating MACE+PME phonons...")
    results_pme = calculate_phonons(nacl_opt_pme, calc_pme, 'mace_pme')
    
    # Step 5: Analysis
    print("\n" + "=" * 60)
    print("STEP 5: Analysis at Γ point")
    print("=" * 60)
    
    analyze_gamma_splitting(results_mace, "MACE only")
    analyze_gamma_splitting(results_pme, "MACE + PME")
    
    # Step 6: Generate plots
    print("\n" + "=" * 60)
    print("STEP 6: Generating comparison plot")
    print("=" * 60)
    
    output_file = plot_comparison(results_mace, results_pme)
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
    
    return output_file


if __name__ == "__main__":
    main()
