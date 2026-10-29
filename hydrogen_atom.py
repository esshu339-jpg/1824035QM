import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path(__file__).parent / "outputs"
OUT.mkdir(exist_ok=True)

def main():
    # Radial s-wave hydrogen atom; distances in Angstrom, energies in eV.
    r = np.linspace(0.001, 25.0, 1800)
    dr = r[1] - r[0]
    potential = -14.3996 / r
    kinetic = 3.80998212  # hbar^2 / (2 m_e), in eV Angstrom^2

    n = len(r) - 2
    diagonal = 2 * kinetic / dr**2 + potential[1:-1]
    off_diagonal = np.full(n - 1, -kinetic / dr**2)
    hamiltonian = np.diag(diagonal) + np.diag(off_diagonal, 1) + np.diag(off_diagonal, -1)

    energies, states = np.linalg.eigh(hamiltonian)
    print("Hydrogen atom (radial s-wave approximation)")
    print(f"Ground-state energy: {energies[0]:.4f} eV")
    print(f"First excited s-state energy: {energies[1]:.4f} eV")

    # Normalize radial wavefunctions for a clearer comparison.
    wavefunctions = states[:, :2].copy()
    for i in range(wavefunctions.shape[1]):
        norm = np.sqrt(np.trapezoid(wavefunctions[:, i]**2, r[1:-1]))
        wavefunctions[:, i] /= norm
    plt.figure(figsize=(9, 5))
    plt.plot(r[1:-1], wavefunctions[:, 0], label=f"Ground state ({energies[0]:.2f} eV)")
    plt.plot(r[1:-1], wavefunctions[:, 1], label=f"First excited s-state ({energies[1]:.2f} eV)")
    plt.xlabel("Radius (Å)")
    plt.ylabel("Normalized radial wavefunction (arb. units)")
    plt.title("Hydrogen Atom: Finite-Difference Solution")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    path = OUT / "hydrogen_atom.png"
    plt.savefig(path, dpi=160)
    plt.close()
    print(f"Plot saved: {path}")

if __name__ == "__main__":
    main()
