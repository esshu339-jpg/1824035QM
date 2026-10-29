import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path(__file__).parent / "outputs"
OUT.mkdir(exist_ok=True)

def main():
    # 1D finite square well. The potential is -V0 inside the well and 0 outside.
    hbar = 1.054571817e-34
    electron_mass = 9.1093837e-31
    electron_charge = 1.602176634e-19
    domain = 8.0e-9
    well_width = 2.0e-9
    well_depth_ev = 10.0
    points = 700

    x = np.linspace(-domain / 2, domain / 2, points)
    dx = x[1] - x[0]
    potential_ev = np.where(np.abs(x) <= well_width / 2, -well_depth_ev, 0.0)
    potential_j = potential_ev * electron_charge
    kinetic = hbar**2 / (2 * electron_mass * dx**2)

    # Boundary values are fixed to zero at the outer edges.
    interior_x = x[1:-1]
    interior_potential = potential_j[1:-1]
    size = len(interior_x)
    hamiltonian = np.diag(2 * kinetic * np.ones(size) + interior_potential)
    hamiltonian += np.diag(-kinetic * np.ones(size - 1), 1)
    hamiltonian += np.diag(-kinetic * np.ones(size - 1), -1)

    energies, states = np.linalg.eigh(hamiltonian)
    energies_ev = energies / electron_charge
    bound_indices = np.where(energies_ev < 0)[0]
    print("Particle in a 1D finite square well")
    print(f"Well depth: {well_depth_ev:.1f} eV; well width: {well_width * 1e9:.1f} nm")
    print(f"Bound states found (E < 0 eV): {len(bound_indices)}")
    for idx in bound_indices[:8]:
        print(f"  State {idx + 1}: {energies_ev[idx]:.4f} eV")

    plt.figure(figsize=(9, 5))
    plt.plot(x * 1e9, potential_ev, color="black", linewidth=2, label="Potential")
    for i in range(min(4, len(energies_ev))):
        state = states[:, i].copy()
        state /= np.max(np.abs(state))
        plt.plot(interior_x * 1e9, energies_ev[i] + 0.45 * state,
                 label=f"State {i+1}: {energies_ev[i]:.2f} eV")
    plt.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    plt.xlabel("Position (nm)")
    plt.ylabel("Energy + scaled wavefunction (eV)")
    plt.title("Particle in a Finite 1D Potential Well")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    path = OUT / "finite_potential_well.png"
    plt.savefig(path, dpi=160)
    plt.close()
    print(f"Plot saved: {path}")

if __name__ == "__main__":
    main()
