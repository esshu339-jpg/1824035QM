import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path(__file__).parent / "outputs"
OUT.mkdir(exist_ok=True)

def main():
    # One-dimensional infinite square well: psi = 0 at x = 0 and x = L.
    hbar = 1.054571817e-34
    electron_mass = 9.1093837e-31
    electron_charge = 1.602176634e-19
    length = 1.0e-9
    points = 500

    x = np.linspace(0, length, points)
    dx = x[1] - x[0]
    kinetic = hbar**2 / (2 * electron_mass * dx**2)
    size = points - 2
    hamiltonian = np.diag(2 * kinetic * np.ones(size))
    hamiltonian += np.diag(-kinetic * np.ones(size - 1), 1)
    hamiltonian += np.diag(-kinetic * np.ones(size - 1), -1)

    energies, states = np.linalg.eigh(hamiltonian)
    energies_ev = energies / electron_charge
    print("Particle in a 1D infinite potential box")
    print("First five energy levels:")
    for i, energy in enumerate(energies_ev[:5], start=1):
        print(f"  n={i}: {energy:.4f} eV")

    plt.figure(figsize=(9, 5))
    for i in range(3):
        state = states[:, i]
        state = state / np.max(np.abs(state))
        plt.plot(x[1:-1] * 1e9, energies_ev[i] + 0.45 * state, label=f"n={i+1}, E={energies_ev[i]:.2f} eV")
    plt.xlabel("Position (nm)")
    plt.ylabel("Energy + scaled wavefunction (eV)")
    plt.title("Particle in an Infinite 1D Box")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    path = OUT / "particle_in_a_box.png"
    plt.savefig(path, dpi=160)
    plt.close()
    print(f"Plot saved: {path}")

if __name__ == "__main__":
    main()
