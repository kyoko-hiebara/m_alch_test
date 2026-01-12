"""
MACE + PME Combined Calculator for ASE
=====================================

Combines MACE with long-range electrostatics via nvalchemiops PME
to investigate LO-TO splitting in ionic systems like NaCl.

NOTE: nvalchemiops PME returns energy in Rydberg units!
      Must convert to eV: E(eV) = E(Ry) × 13.606
"""

import torch
import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from typing import Dict, Optional

# Unit conversion: Rydberg to eV
RY_TO_EV = 13.606


class MACEPMECalculator(Calculator):
    """
    ASE Calculator combining MACE with Particle Mesh Ewald electrostatics.
    
    E_total = E_MACE + E_PME
    F_total = F_MACE + F_PME
    """
    
    implemented_properties = ['energy', 'forces', 'stress']
    
    def __init__(
        self,
        mace_model: str = "medium-omat-0",
        charges: Optional[Dict[str, float]] = None,
        pme_cutoff: float = 12.0,
        pme_accuracy: float = 1e-6,
        device: str = 'cuda',
        default_dtype: str = 'float32',
        verbose: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.device = device
        self.pme_cutoff = pme_cutoff
        self.pme_accuracy = pme_accuracy
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
        
        # Convert from Rydberg to eV
        # nvalchemiops PME returns energy in Rydberg units!
        energy = atom_energies.sum().item() * RY_TO_EV
        forces = atom_forces.detach().cpu().numpy() * RY_TO_EV  # Force also in Ry/Å → eV/Å
        
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
