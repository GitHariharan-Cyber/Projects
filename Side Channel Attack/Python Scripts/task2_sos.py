import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ===== CONFIGURATION =====
CSV_FILE_PATH = "C1--pt--00003.csv"         # Power trace CSV file
CSV_SKIP_HEADER = 5                         # Number of header rows to skip
TRACE_START_INDEX = 2627250                 # Starting sample index for processing
SAMPLES_PER_CLOCK = 1250                    # Samples in one clock cycle
CLOCKS_PER_SLOT = 54                        # Number of cycles per slot
TOTAL_SLOTS_TO_PROCESS = 230                # Total slots to extract
OUTPUT_DIRECTORY = Path("unique_results")   # Directory for output files
OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

# True scalar key (ground truth) given in hexadecimal
SCALAR_KEY_HEX = "181856adc1e7df1378491fa736f2d02e8acf1b9425eb2b061ff0e9e8246"


# ===== HELPER FUNCTIONS =====

def load_trace_from_csv(file_path, skip_header=5):
    """
    Load power trace from CSV file.
    Returns only the amplitude column as numpy array.
    """
    df = pd.read_csv(file_path, skiprows=skip_header, names=["Time", "Amplitude"])
    return df["Amplitude"].values


def extract_trace_slots(trace_data, start_index, samples_per_cycle, cycles_per_slot, total_slots):
    """
    Extract slots from raw trace.
    Reshape into a 3D array: [slots, cycles, samples_per_cycle]
    """
    total_samples = total_slots * cycles_per_slot * samples_per_cycle
    clipped_data = trace_data[start_index : start_index + total_samples]
    return clipped_data.reshape(total_slots, cycles_per_slot, samples_per_cycle)


def generate_key_candidates(slot_matrix):
    """
    Generate candidate key bits using 'sum of squared samples' compression.
    Steps:
      - Compute energy = sum of squares per cycle.
      - Compute mean energy across slots for each cycle.
      - Compare each cycle’s energy with mean → candidate bits.
  
    """
    cycle_energy_matrix = np.sum(np.square(slot_matrix), axis=2)       
    mean_cycle_energy = np.mean(cycle_energy_matrix, axis=0)         
    candidate_bits = (cycle_energy_matrix <= mean_cycle_energy).astype(np.int8).T  
    return candidate_bits, mean_cycle_energy, cycle_energy_matrix


def convert_scalar_to_bits(hex_string, num_bits, skip_bits=8):
    """
    Convert scalar key from hex string to binary array.
    Skips the first `skip_bits` (implementation detail).
    """
    binary_string = bin(int(hex_string, 16))[2:].zfill(len(hex_string) * 4)
    return np.array([int(b) for b in binary_string[skip_bits : skip_bits + num_bits]], dtype=int)


def compare_candidates_with_scalar(candidates, true_scalar_bits):
    """
    Evaluate correctness of candidate bits vs. true scalar bits.
    Computes:
      - Direct match
      - Left-shifted match
      - Right-shifted match
      - Best match (The best among the deltas)
    """
    scalar_left_shift = true_scalar_bits[1:]
    scalar_right_shift = true_scalar_bits[:-1]

    delta_direct_all = []
    delta_left_all = []
    delta_right_all = []
    delta_best_all = []

    for cycle_index, candidate_bits in enumerate(candidates):
        # Skip first 2 bits (implementation detail)
        candidate_bits = candidate_bits[2:]

        # Direct match
        delta_direct = (np.sum(candidate_bits == true_scalar_bits) / len(true_scalar_bits)) * 100

        # Left shift
        delta_left = (np.sum(candidate_bits[:-1] == scalar_left_shift) / (len(true_scalar_bits) - 1)) * 100

        # Right shift
        delta_right = (np.sum(candidate_bits[1:] == scalar_right_shift) / (len(true_scalar_bits) - 1)) * 100

        # Best (choose farthest from random guessing 50%)
        options = [delta_direct, delta_left, delta_right]
        best_value = options[np.argmax([abs(x - 50.0) for x in options])]

        # Store results
        delta_direct_all.append(delta_direct)
        delta_left_all.append(delta_left)
        delta_right_all.append(delta_right)
        delta_best_all.append(best_value)

        print(f"Cycle {cycle_index:02d}: Direct={delta_direct:.2f}%  Left={delta_left:.2f}%  Right={delta_right:.2f}%  -> Best={best_value:.2f}%")

    return (
        np.array(delta_direct_all),
        np.array(delta_left_all),
        np.array(delta_right_all),
        np.array(delta_best_all)
    )


# ===== MAIN PIPELINE =====

def main():
    print("=== Starting Scalar Recovery Analysis ===")

    # Load power trace
    raw_trace = load_trace_from_csv(CSV_FILE_PATH, CSV_SKIP_HEADER)

    # Cut and reshape into slots
    slot_matrix = extract_trace_slots(raw_trace, TRACE_START_INDEX, SAMPLES_PER_CLOCK, CLOCKS_PER_SLOT, TOTAL_SLOTS_TO_PROCESS)

    # Generate candidate key bits
    key_candidates, mean_cycle_energy, cycle_energy_matrix = generate_key_candidates(slot_matrix)
    print(f"Generated candidate key bits: {key_candidates.shape} (cycles × slots)")

    # Ground truth scalar bits
    true_scalar_bits = convert_scalar_to_bits(SCALAR_KEY_HEX, TOTAL_SLOTS_TO_PROCESS, skip_bits=8)

    # Evaluate correctness
    delta_direct, delta_left, delta_right, delta_best = compare_candidates_with_scalar(key_candidates, true_scalar_bits)

    # === Plot results ===
    plt.figure(figsize=(12, 6))
    plt.plot(delta_direct, "o-", label="δ Direct Match", linewidth=2, markersize=5)
    plt.plot(delta_left, "s--", label="δ Left Shift", linewidth=2, markersize=5)
    plt.plot(delta_right, "d-.", label="δ Right Shift", linewidth=2, markersize=5)
    plt.plot(delta_best, "x-", label="δ Best Match", linewidth=2, markersize=5, color="red")

    plt.xlabel("Cycle index (0–53)")
    plt.ylabel("Correctness (%)")
    plt.title("Comparison-to-Mean Scalar Recovery (per cycle) using SOS compression method for EM trace")
    plt.legend()
    plt.grid(True, linestyle=":")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIRECTORY / "delta_comparison_plot.png")
    plt.show()

    # Save outputs
    np.savetxt(OUTPUT_DIRECTORY / "candidate_bits.txt", key_candidates, fmt="%d")
    np.savetxt(OUTPUT_DIRECTORY / "delta_direct.txt", delta_direct, fmt="%.6f")
    np.savetxt(OUTPUT_DIRECTORY / "delta_left.txt", delta_left, fmt="%.6f")
    np.savetxt(OUTPUT_DIRECTORY / "delta_right.txt", delta_right, fmt="%.6f")
    np.savetxt(OUTPUT_DIRECTORY / "delta_best.txt", delta_best, fmt="%.6f")
    np.savetxt(OUTPUT_DIRECTORY / "cycle_energy_matrix.txt", cycle_energy_matrix, fmt="%.6f")

    print("\nSaved analysis results in:", OUTPUT_DIRECTORY)


if __name__ == "__main__":
    main()
