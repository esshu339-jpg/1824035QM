"""
Unit 3 (Q2): s-wave Schrodinger equation for the screened Coulomb potential

    V(r) = -(e^2 / r) * exp(-r / a)

Method: finite difference.
    u''(r) = A(r) u(r),   A(r) = (2m / hbar^2) [V(r) - E]
Writing u'' as (u[i+1] - 2u[i] + u[i-1]) / h^2 turns the equation into a
matrix eigenvalue problem H u = E u, where H is tridiagonal:

    H[i,i]   = 2*C/h^2 + V(r_i)
    H[i,i+1] = H[i,i-1] = -C/h^2          with C = hbar^2 / 2m

Boundary conditions: u(0) = 0 and u(r_max) = 0.

Units: energy in eV, length in Angstrom
    e^2 = 3.795^2 (eV A),  hbar*c = 1973 eV A,  m c^2 = 0.511e6 eV
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh_tridiagonal

# ---------------- constants ----------------
e2 = 3.795 ** 2          # eV A
hc = 1973.0              # eV A
mc2 = 0.511e6            # eV
C = hc ** 2 / (2 * mc2)  # hbar^2 / 2m in eV A^2


def potential(r, a):
    """Screened Coulomb potential in eV."""
    return -(e2 / r) * np.exp(-r / a)


def ground_state(a, r_max=60.0, N=6000):
    """
    Returns the ground state energy (eV), the grid r and the
    normalised wavefunction u(r) for a given screening length a.
    """
    h = r_max / (N + 1)
    r = h * np.arange(1, N + 1)          # interior points only, u(0)=u(rmax)=0

    diag = 2 * C / h ** 2 + potential(r, a)
    off = -C / h ** 2 * np.ones(N - 1)

    # we only need the lowest eigenvalue
    E, vec = eigh_tridiagonal(diag, off, select='i', select_range=(0, 0))
    u = vec[:, 0]

    # normalise so that integral of u^2 dr = 1 (trapezoid rule)
    u = u / np.sqrt(np.trapezoid(u ** 2, r))
    if u[np.argmax(np.abs(u))] < 0:      # fix the overall sign
        u = -u
    return E[0], r, u


if __name__ == "__main__":
    a_values = [3.0, 5.0, 7.0]

    # ---- convergence check: change step size and see if E settles ----
    print("Convergence check for a = 5 A")
    print("   N      E0 (eV)")
    for N in [1000, 2000, 4000, 8000]:
        E, _, _ = ground_state(5.0, N=N)
        print(f"{N:6d}   {E:10.5f}")
    print()

    # ---- main results ----
    plt.figure(figsize=(7, 5))
    print("Ground state energies (screened Coulomb)")
    for a in a_values:
        E, r, u = ground_state(a)
        print(f"a = {a:.0f} A  ->  E0 = {E:.3f} eV")
        plt.plot(r, u, label=f"a = {a:.0f} Å,  E0 = {E:.3f} eV")

    plt.xlim(0, 25)
    plt.xlabel("r (Å)")
    plt.ylabel("u(r)")
    plt.title("Ground state wavefunction, screened Coulomb potential")
    plt.axhline(0, color="k", lw=0.5)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("screened_coulomb_wavefunction.png", dpi=200)
    plt.show()
