import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ===== CONFIGURATION =====
TRACE_FILE = "PT_2.5Gs_NoFilter.csv"
REF_FILE = "PT_2_5Gs_NoFilter_csv_StringArray_startPoint_65570results_0_100.txt"
SKIP_LINES = 5
TRACE_START = 65565
SAMPLES_PER_CYCLE = 625
CYCLES_PER_SLOT = 54
TOTAL_SLOTS = 230
OUTPUT_DIR = Path("analysis_out_abs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SCALAR_HEX = "93919255FD4359F4C2B67DEA456EF70A545A9C44D46F7F409F96CB52CC"

# ===== FUNCTIONS =====

def load_power_trace(file_path, skip=5):
    """Load a 1D array of power amplitudes from CSV trace file."""
    df = pd.read_csv(file_path, skiprows=skip, names=["Time", "Amplitude"])
    return df["Amplitude"].values

def segment_trace(trace_array, start_idx, samples_per_cycle, cycles_per_slot, num_slots):
    """Extract required samples and reshape into (slots × cycles × samples)."""
    total_samples = num_slots * cycles_per_slot * samples_per_cycle
    selected = trace_array[start_idx : start_idx + total_samples]
    return selected.reshape(num_slots, cycles_per_slot, samples_per_cycle)

def generate_key_candidates_abs(trace_cube):
    """
    Compute per-cycle sum of absolute values and generate candidate bits.
   
    """
    cycle_energy_matrix = np.sum(np.abs(trace_cube), axis=2)  # shape: (slots × cycles)
    mean_energy_per_cycle = np.mean(cycle_energy_matrix, axis=0)
    candidates = (cycle_energy_matrix <= mean_energy_per_cycle).astype(np.int8).T
    return candidates, mean_energy_per_cycle, cycle_energy_matrix

def hex_to_bitarray(hex_str, n_bits, skip_bits=2):
    """Convert hex scalar to binary array, skipping first 'skip_bits' bits."""
    bin_str = bin(int(hex_str, 16))[2:].zfill(len(hex_str)*4)
    return np.array([int(b) for b in bin_str[skip_bits : skip_bits+n_bits]], dtype=int)

def compare_candidates(candidates, scalar_bits_array):
    """
    Evaluate each candidate against true scalar.
    Returns:
      - best delta per candidate
      - all delta results (direct, left-shifted, right-shifted)
    """
    shift_left  = scalar_bits_array[1:]
    shift_right = scalar_bits_array[:-1]

    best_deltas = []
    all_deltas = []

    for idx, cand in enumerate(candidates):
        d_direct = (np.sum(cand == scalar_bits_array) / len(scalar_bits_array)) * 100
        d_left   = (np.sum(cand[:-1] == shift_left) / (len(scalar_bits_array)-1)) * 100
        d_right  = (np.sum(cand[1:] == shift_right) / (len(scalar_bits_array)-1)) * 100

        deltas = [d_direct, d_left, d_right]
        labels = ["Direct", "LeftShift", "RightShift"]
        best_idx = np.argmax([abs(x - 50.0) for x in deltas])

        best_deltas.append(deltas[best_idx])
        all_deltas.append(tuple(deltas))

        print(f"Candidate {idx:02d}: Direct={d_direct:.2f}, Left={d_left:.2f}, Right={d_right:.2f} -> Best={labels[best_idx]}")

    return np.array(best_deltas), all_deltas

def load_reference_values(file_path):
    """Load professor reference delta values from a text file."""
    with open(file_path, "r") as f:
        values = [float(line.strip()) for line in f if line.strip()]
    return np.array(values)

# ===== MAIN EXECUTION =====
def main():
    print("=== Power Trace Key Analysis (Sum of Absolute Values) ===")

    # Load and reshape trace
    raw_trace = load_power_trace(TRACE_FILE, SKIP_LINES)
    trace_cube = segment_trace(raw_trace, TRACE_START, SAMPLES_PER_CYCLE, CYCLES_PER_SLOT, TOTAL_SLOTS)

    # Generate candidates (absolute values compression)
    key_candidates, mean_energy, cycle_energy_matrix = generate_key_candidates_abs(trace_cube)
    print(f"Candidates shape: {key_candidates.shape} (cycles × slots)")

    # Convert scalar to bit array
    scalar_bits_array = hex_to_bitarray(SCALAR_HEX, TOTAL_SLOTS, skip_bits=2)

    # Evaluate candidates
    best_deltas, all_deltas = compare_candidates(key_candidates, scalar_bits_array)

    # Load reference values
    ref_deltas = load_reference_values(REF_FILE)

    # Save outputs
    np.savetxt(OUTPUT_DIR / "key_candidates_bits_abs.txt", key_candidates, fmt="%d")
    np.savetxt(OUTPUT_DIR / "best_deltas_abs.txt", best_deltas, fmt="%.6f")
    np.savetxt(OUTPUT_DIR / "cycle_energy_matrix_abs.txt", cycle_energy_matrix, fmt="%.6f")

    # Plot comparison
    plt.figure(figsize=(12, 6))
    plt.plot(best_deltas, "o-", label="Delta Best (computed)", linewidth=2, markersize=5, color="darkorange")
    plt.plot(ref_deltas, "s--", label="Reference δ-values", alpha=0.7, color="black")
    plt.xlabel("Key Candidate Index")
    plt.ylabel("Correctness (%)")
    plt.title("Candidate Correctness - Sum of Absolute Values Compression")
    plt.legend()
    plt.grid(True, linestyle=":")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "delta_best_comparison_abs.png")
    plt.show()

    print("\nSaved files in:", OUTPUT_DIR)
    print("  - key_candidates_bits_abs.txt")
    print("  - best_deltas_abs.txt")
    print("  - cycle_energy_matrix_abs.txt")
    print("  - delta_best_comparison_abs.png")

if __name__ == "__main__":
    main()
