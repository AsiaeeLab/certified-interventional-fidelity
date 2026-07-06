"""Master script: run all CIF experiments and generate figures."""

from __future__ import annotations

import time


def main() -> None:
    start = time.time()
    print("=" * 60)
    print("CIF Experiment Suite")
    print("=" * 60)

    # Step 1: Run E1
    print("\n[E1] Certified IIA for MNIST abstractions...")
    from e1_certified_iia import run_e1

    run_e1()

    # Step 2: Run E2
    print("\n[E2] Certified patching in GPT-2 Small (IOI)...")
    from e2_gpt2_patching import run_e2

    run_e2()

    # Step 3: Run E3
    print("\n[E3] Sensitivity to Pi... (skipped; using existing results)")

    # Step 4: Run E4
    print("\n[E4] Coverage sanity check...")
    from e4_coverage import run_e4

    run_e4()

    # Step 5: Generate all figures
    print("\n[Plots] Generating all figures...")
    from plot_all import make_all_plots

    make_all_plots()

    elapsed = time.time() - start
    print(f"\nAll done in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
