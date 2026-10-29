# Quantum Mechanics Simulations (Finite Difference Method)

A small Python project that numerically explores three introductory quantum-mechanics systems:
1. Hydrogen atom (radial, s-wave approximation)
2. Particle in an infinite 1D box
3. Particle in a finite 1D potential well

The programs use the finite-difference method to build a Hamiltonian matrix, solve its eigenvalue problem, print energy levels, and save plots to `outputs/`.

## Requirements
- Python 3.9+
- NumPy
- Matplotlib

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate        # Linux/macOS
pip install -r requirements.txt
```

On Kali Linux, if virtual-environment creation or pip is unavailable, install `python3-venv` first:
```bash
sudo apt update
sudo apt install python3-venv python3-pip
```

## Run
Run each simulation separately:
```bash
python3 hydrogen_atom.py
python3 particle_in_a_box.py
python3 finite_potential_well.py
```

Or run all three:
```bash
python3 main.py
```

Plots are saved in `outputs/` and can also open in a graphical window when a desktop display is available. In headless environments, the scripts still save the PNG files.

## Model notes
- The hydrogen calculation uses the radial Schrödinger equation for the s-wave (`l = 0`) and energies are expected to approach the familiar ground-state value near -13.6 eV as the grid/domain are refined. The radial wavefunction is plotted after normalization.
- The infinite-box calculation uses zero wavefunction at both boundaries.
- The finite well is a 1D square well with a configurable depth and width. It is a different model from the infinite box; the original uploaded file was a duplicate of the infinite-box calculation, so it has been corrected here.
- This is an educational numerical model, not a precision scientific package.
