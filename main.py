from hydrogen_atom import main as run_hydrogen
from particle_in_a_box import main as run_box
from finite_potential_well import main as run_finite_well

def main():
    print("=" * 60)
    run_hydrogen()
    print("\n" + "=" * 60)
    run_box()
    print("\n" + "=" * 60)
    run_finite_well()
    print("\nAll simulations completed. Check the outputs/ folder for plots.")

if __name__ == "__main__":
    main()
