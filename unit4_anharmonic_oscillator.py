"""
Unit 4 (Q1): s-wave Schrodinger equation for the anharmonic oscillator

    V(r) = (1/2) k r^2 + (1/3) b r^3

Method: finite difference (same idea as in unit 3).
    -(hbar^2 / 2m) u'' + V(r) u = E u
    u'' ~ (u[i+1] - 2u[i] + u[i-1]) / h^2
gives a tridiagonal matrix H whose lowest eigenvalue is the ground state
energy. Boundary conditions: u(0) = 0 and u(r_max) = 0.

Units: energy in MeV, length in fm
    m c^2 = 940 MeV,  k = 100 MeV fm^-2,  hbar*c = 197.3 MeV fm
    b = 0, 10, 30 MeV fm^-3
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh_tridiagonal

# ---------------- constants ----------------
mc2 = 940.0              # MeV
k = 100.0                # MeV fm^-2
hc = 197.3               # MeV fm
C = hc ** 2 / (2 * mc2)  # hbar^2 / 2m in MeV fm^2


def potential(r, b):
    """Anharmonic potential in MeV."""
    return 0.5 * k * r ** 2 + b * r ** 3 / 3.0


def ground_state(b, r_max=10.0, N=4000):
    """
    Ground state energy (MeV), grid and normalised wavefunction for a
    given cubic coefficient b.
    """
    h = r_max / (N + 1)
    r = h * np.arange(1, N + 1)

    diag = 2 * C / h ** 2 + potential(r, b)
    off = -C / h ** 2 * np.ones(N - 1)

    E, vec = eigh_tridiagonal(diag, off, select='i', select_range=(0, 0))
    u = vec[:, 0]

    u = u / np.sqrt(np.trapezoid(u ** 2, r))   # normalise
    if u[np.argmax(np.abs(u))] < 0:
        u = -u
    return E[0], r, u


if __name__ == "__main__":
    b_values = [0.0, 10.0, 30.0]

    # ---- check for b = 0: exact answer is (3/2) hbar*omega ----
    omega_term = hc * np.sqrt(k / mc2)          # hbar*omega in MeV
    print(f"Analytical result for b = 0 : E0 = 1.5*hbar*omega = {1.5 * omega_term:.3f} MeV")
    print()

    # ---- convergence check ----
    print("Convergence check for b = 10")
    print("   N      E0 (MeV)")
    for N in [500, 1000, 2000, 4000]:
        E, _, _ = ground_state(10.0, N=N)
        print(f"{N:6d}   {E:10.5f}")
    print()

    # ---- main results ----
    plt.figure(figsize=(7, 5))
    print("Ground state energies (anharmonic oscillator)")
    for b in b_values:
        E, r, u = ground_state(b)
        print(f"b = {b:4.0f} MeV fm^-3  ->  E0 = {E:.3f} MeV")
        plt.plot(r, u, label=f"b = {b:.0f},  E0 = {E:.2f} MeV")

    plt.xlim(0, 4)
    plt.xlabel("r (fm)")
    plt.ylabel("u(r)")
    plt.title("Ground state wavefunction, anharmonic oscillator")
    plt.axhline(0, color="k", lw=0.5)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("anharmonic_wavefunction.png", dpi=200)
    plt.show()
