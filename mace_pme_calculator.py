"""
MACE + PME Combined Calculator for ASE
=====================================

Combines MACE with long-range electrostatics via nvalchemiops PME
to investigate LO-TO splitting in ionic systems like NaCl.

NOTE: nvalchemiops PME assumes k_e = 1 (atomic units).
      For eV-Å system, multiply by Coulomb constant:
      k_e = e²/(4πε₀) = 14.3996 eV·Å
"""

import torch
import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from typing import Dict, Optional

# Coulomb constant for eV-Å system
# k_e = e²/(4πε₀) in eV·Å
COULOMB_CONST = 14.3996


class MACEPMECalculator(Calculator):
    """
    ASE Calculator combining MACE with Particle Mesh Ewald electrostatics.
    
    E_total = E_MACE + pme_scale * E_PME
    F_total = F_MACE + pme_scale * F_PME
    
    Note on double counting:
    MACE implicitly learns short-range Coulomb from DFT training data.
    Adding full PME causes double counting of short-range part.
    Use pme_scale < 1.0 to mitigate, or ideally subtract short-range PME.
    """
    
    implemented_properties = ['energy', 'forces', 'stress']
    
    def __init__(
        self,
        mace_model: str = "medium-omat-0",
        charges: Optional[Dict[str, float]] = None,
        pme_cutoff: float = 12.0,
        pme_accuracy: float = 1e-6,
        pme_scale: float = 1.0,
        device: str = 'cuda',
        default_dtype: str = 'float32',
        verbose: bool = False,
        **kwargs
    ):
        """
        Parameters
        ----------
        pme_scale : float
            Scaling factor for PME contribution (0-1).
            Use < 1.0 to reduce double counting with MACE.
            Default 1.0 = full PME.
        """
        super().__init__(**kwargs)
        
        self.device = device
        self.pme_cutoff = pme_cutoff
        self.pme_accuracy = pme_accuracy
        self.pme_scale = pme_scale
        self.charges_dict = charges or {}
        self.verbose = verbose
        
        if default_dtype == 'float32':
            torch.set_default_dtype(torch.float32)
        else:
            torch.set_default_dtype(torch.float64)
        
        self._init_mace(mace_model)
        
    def _init_mace(self, mace_model: str):
        """Initialize MACE calculator"""
        try:
            from mace.calculators import MACECalculator
            self.mace = MACECalculator(
                model_paths=mace_model,
                device=self.device,
                default_dtype='float32'
            )
        except Exception as e:
            print(f"Warning: Could not load MACE model '{mace_model}': {e}")
            print("Falling back to MACE-MP-0 small")
            from mace.calculators import mace_mp
            self.mace = mace_mp(model='small', device=self.device)
    
    def _get_charges(self, atoms) -> torch.Tensor:
        """Get charge tensor from atoms"""
        charges = []
        for symbol in atoms.get_chemical_symbols():
            if symbol in self.charges_dict:
                charges.append(self.charges_dict[symbol])
            else:
                charges.append(0.0)
        return torch.tensor(charges, dtype=torch.float32, device=self.device)
    
    def _compute_pme(self, atoms) -> tuple:
        """Compute PME electrostatic energy and forces using nvalchemiops."""
        from nvalchemiops.neighborlist import neighbor_list
        from nvalchemiops.interactions.electrostatics import particle_mesh_ewald
        
        # Prepare inputs
        positions = torch.tensor(
            atoms.positions, 
            dtype=torch.float32, 
            device=self.device,
            requires_grad=True
        )
        
        # Cell needs shape [num_systems, 3, 3]
        cell = torch.tensor(
            atoms.cell.array, 
            dtype=torch.float32, 
            device=self.device
        ).unsqueeze(0)
        
        # PBC
        pbc = torch.tensor(atoms.pbc, dtype=torch.bool, device=self.device)
        
        # Charges
        charges = self._get_charges(atoms)
        
        # Skip if no charges or all zero
        if charges.abs().sum() < 1e-10:
            return 0.0, np.zeros((len(atoms), 3))
        
        # Build neighbor list for real-space PME part
        neighbor_matrix, num_neighbors, shift_matrix = neighbor_list(
            positions,
            cutoff=self.pme_cutoff,
            cell=cell,
            pbc=pbc,
            method="cell_list"
        )
        
        # Compute PME with forces
        pme_result = particle_mesh_ewald(
            positions=positions,
            charges=charges,
            cell=cell,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=shift_matrix,
            accuracy=self.pme_accuracy,
            compute_forces=True
        )
        
        # With compute_forces=True, returns (energies, forces)
        atom_energies, atom_forces = pme_result
        
        # Convert from atomic units (k_e=1) to eV-Å system
        # Multiply by Coulomb constant k_e = 14.3996 eV·Å
        energy = atom_energies.sum().item() * COULOMB_CONST
        forces = atom_forces.detach().cpu().numpy() * COULOMB_CONST
        
        return energy, forces
    
    def calculate(
        self, 
        atoms=None, 
        properties=['energy', 'forces'],
        system_changes=all_changes
    ):
        """Calculate energy and forces"""
        
        super().calculate(atoms, properties, system_changes)
        
        # Get MACE results
        self.mace.calculate(atoms, properties)
        e_mace = self.mace.results['energy']
        f_mace = self.mace.results['forces']
        
        # Get PME results
        if self.charges_dict:
            e_pme, f_pme = self._compute_pme(atoms)
            # Apply scaling factor
            e_pme *= self.pme_scale
            f_pme *= self.pme_scale
        else:
            e_pme = 0.0
            f_pme = np.zeros_like(f_mace)
        
        # Combine
        self.results['energy'] = e_mace + e_pme
        self.results['forces'] = f_mace + f_pme
        
        # Store components for analysis
        self.results['energy_mace'] = e_mace
        self.results['energy_pme'] = e_pme
        
        # Verbose output
        if self.verbose:
            print(f"[MACE+PME] E_total={e_mace + e_pme:.6f} eV "
                  f"(MACE={e_mace:.6f}, PME={e_pme:.6f})")
        
        if 'stress' in properties and 'stress' in self.mace.results:
            self.results['stress'] = self.mace.results['stress']


class MACEOnlyCalculator(Calculator):
    """Wrapper for MACE-only calculations (for comparison)."""
    
    implemented_properties = ['energy', 'forces', 'stress']
    
    def __init__(self, mace_model: str = "medium-omat-0", device: str = 'cuda', **kwargs):
        super().__init__(**kwargs)
        self.device = device
        self._init_mace(mace_model)
        
    def _init_mace(self, mace_model: str):
        try:
            from mace.calculators import MACECalculator
            self.mace = MACECalculator(
                model_paths=mace_model,
                device=self.device,
                default_dtype='float32'
            )
        except:
            from mace.calculators import mace_mp
            self.mace = mace_mp(model='small', device=self.device)
    
    def calculate(self, atoms=None, properties=['energy', 'forces'], system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.mace.calculate(atoms, properties)
        self.results = dict(self.mace.results)
