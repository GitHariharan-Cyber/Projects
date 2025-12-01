#!/usr/bin/env python3
"""
Simple Mean Comparison Attack with Visualization
- Uses only comparison to mean ,Selects fixed regions based on signal characteristics
- Creates overlay plots of all slots
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------- CONFIGURATION ----------------
CSV_PATH = Path("C1--pt--00003.csv")
COL_IDX = 1
START_IDX = 2627250
NUM_SLOTS = 230
CYCLES_PER_SLOT = 54
SAMPLES_PER_CYCLE = 1250
SLOT_LEN = CYCLES_PER_SLOT * SAMPLES_PER_CYCLE

REAL_KEY_HEX = "1856adc1e7df1378491fa736f2d02e8acf1b9425eb2b061ff0e9e8246"
OUT_DIR = Path("Results_Task3")
# -----------------------------------------------

OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_trace_data(path, col_idx):
    """Load trace data from CSV."""
    try:
        df = pd.read_csv(path, skiprows=5, header=None)
    except:
        df = pd.read_csv(path, header=5)
    return df.iloc[:, col_idx].astype(np.float64).values

def hex_to_bit_array(hex_string, bit_length):
    """Convert hexadecimal to bit array."""
    hex_clean = hex_string.strip().lower()
    integer_val = int(hex_clean, 16)
    bit_string = bin(integer_val)[2:].zfill(bit_length)
    return np.array([int(bit) for bit in bit_string[-bit_length:]])

def mean_comparison_attack(cycle_data, use_full_cycle=True, region_start=0, region_end=None):
    """
    Simple mean comparison attack.
    If use_full_cycle=True: uses entire cycle
    If use_full_cycle=False: uses specified region
    """
    if use_full_cycle:
        # Use entire cycle for comparison
        region_data = cycle_data
    else:
        # Use specified region
        if region_end is None:
            region_end = cycle_data.shape[1]
        region_data = cycle_data[:, region_start:region_end]
    
    # Calculate mean for each slot
    slot_means = np.mean(region_data, axis=1)
    
    # Use median as threshold
    threshold = np.median(slot_means)
    
    # Assign bits: above threshold = 1, below threshold = 0
    candidate_bits = (slot_means > threshold).astype(int)
    
    return candidate_bits

def evaluate_candidate(candidate_bits, reference_bits):
    """Evaluate candidate bits against reference."""
    candidate_228 = candidate_bits[2:230]  # Skip first 2 slots
    
    direct_accuracy = np.mean(candidate_228 == reference_bits)
    inverted_accuracy = np.mean(candidate_228 != reference_bits)
    
    best_accuracy = max(direct_accuracy, inverted_accuracy)
    return best_accuracy, direct_accuracy, inverted_accuracy

def create_overlay_plots(slot_data, reference_bits, output_dir):
    """Create overlay plots of all 230 slots."""
    
    # Reshape slot data to 2D: (NUM_SLOTS, SLOT_LEN)
    slots_2d = slot_data.reshape(NUM_SLOTS, SLOT_LEN)
    x_axis = np.arange(SLOT_LEN)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # Plot 1: All slots overlay (black lines)
    print("Creating black overlay plot...")
    for slot_idx in range(NUM_SLOTS):
        ax1.plot(x_axis, slots_2d[slot_idx], color='black', alpha=0.1, linewidth=0.5)
    
    ax1.set_title('Overlay of All 230 Slots (Black Lines)\nNo Knowledge of Real Key', 
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel('Sample Index')
    ax1.set_ylabel('Amplitude')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, SLOT_LEN)
    
    # Plot 2: Slots colored by real key bits (blue for 0, orange for 1)
    print("Creating colored overlay plot...")
    colors = ['blue' if bit == 0 else 'orange' for bit in reference_bits]
    
    for slot_idx in range(NUM_SLOTS):
        # Use the corresponding color for each slot based on real key
        ax2.plot(x_axis, slots_2d[slot_idx], color=colors[slot_idx], alpha=0.2, linewidth=0.5)
    
    ax2.set_title('Overlay of All 230 Slots Colored by Real Key Bits\nBlue = Bit 0, Orange = Bit 1', 
                  fontsize=14, fontweight='bold')
    ax2.set_xlabel('Sample Index')
    ax2.set_ylabel('Amplitude')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, SLOT_LEN)
    
    # Add legend for the colored plot
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='blue', alpha=0.7, label='Bit 0 (Real Key)'),
        Patch(facecolor='orange', alpha=0.7, label='Bit 1 (Real Key)')
    ]
    ax2.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    
    # Save the figure
    plot_path = output_dir / "all_slots_overlay_comparison.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[+] Overlay plots saved to: {plot_path}")
    return plot_path

def main():
    print("Loading trace data...")
    trace_data = load_trace_data(CSV_PATH, COL_IDX)
    
    # Validate trace length
    required_length = START_IDX + NUM_SLOTS * SLOT_LEN
    if len(trace_data) < required_length:
        raise ValueError(f"Trace too short: {len(trace_data)} < {required_length}")
    
    # Extract slot data
    trace_segment = trace_data[START_IDX:START_IDX + NUM_SLOTS * SLOT_LEN]
    slot_data = trace_segment.reshape(NUM_SLOTS, CYCLES_PER_SLOT, SAMPLES_PER_CYCLE)
    
    # Convert reference key
    reference_bits = hex_to_bit_array(REAL_KEY_HEX, 228)
    # Pad with 2 zeros at the beginning to match 230 slots
    reference_bits_full = np.concatenate([np.array([0, 0]), reference_bits])
    
    print("Creating overlay plots of all slots...")
    # Create the overlay plots
    plot_path = create_overlay_plots(slot_data, reference_bits_full, OUT_DIR)
    
    print("Testing cycles with mean comparison...")
    results = []
    
    # Test each cycle with different approaches
    for cycle in range(CYCLES_PER_SLOT):
        cycle_data = slot_data[:, cycle, :]
        
        # Approach 1: Use full cycle
        candidate_bits = mean_comparison_attack(cycle_data, use_full_cycle=True)
        best_acc, direct_acc, inverted_acc = evaluate_candidate(candidate_bits, reference_bits)
        
        results.append({
            'cycle': cycle,
            'method': 'full_cycle',
            'start': 0,
            'end': SAMPLES_PER_CYCLE,
            'best_accuracy': best_acc,
            'direct_accuracy': direct_acc,
            'inverted_accuracy': inverted_acc,
            'candidate_bits': candidate_bits
        })
        
        # Approach 2: Use first half of cycle
        candidate_bits = mean_comparison_attack(cycle_data, use_full_cycle=False, 
                                              region_start=0, region_end=SAMPLES_PER_CYCLE//2)
        best_acc, direct_acc, inverted_acc = evaluate_candidate(candidate_bits, reference_bits)
        
        results.append({
            'cycle': cycle,
            'method': 'first_half',
            'start': 0,
            'end': SAMPLES_PER_CYCLE//2,
            'best_accuracy': best_acc,
            'direct_accuracy': direct_acc,
            'inverted_accuracy': inverted_acc,
            'candidate_bits': candidate_bits
        })
        
        # Approach 3: Use second half of cycle
        candidate_bits = mean_comparison_attack(cycle_data, use_full_cycle=False,
                                              region_start=SAMPLES_PER_CYCLE//2, 
                                              region_end=SAMPLES_PER_CYCLE)
        best_acc, direct_acc, inverted_acc = evaluate_candidate(candidate_bits, reference_bits)
        
        results.append({
            'cycle': cycle,
            'method': 'second_half',
            'start': SAMPLES_PER_CYCLE//2,
            'end': SAMPLES_PER_CYCLE,
            'best_accuracy': best_acc,
            'direct_accuracy': direct_acc,
            'inverted_accuracy': inverted_acc,
            'candidate_bits': candidate_bits
        })
        
        # Approach 4: Use middle section
        quarter = SAMPLES_PER_CYCLE // 4
        candidate_bits = mean_comparison_attack(cycle_data, use_full_cycle=False,
                                              region_start=quarter, 
                                              region_end=3*quarter)
        best_acc, direct_acc, inverted_acc = evaluate_candidate(candidate_bits, reference_bits)
        
        results.append({
            'cycle': cycle,
            'method': 'middle_section',
            'start': quarter,
            'end': 3*quarter,
            'best_accuracy': best_acc,
            'direct_accuracy': direct_acc,
            'inverted_accuracy': inverted_acc,
            'candidate_bits': candidate_bits
        })
    
    # Sort by accuracy
    results.sort(key=lambda x: x['best_accuracy'], reverse=True)
    
    print(f"\n{'='*70}")
    print("TOP 10 RESULTS (Simple Mean Comparison)")
    print(f"{'='*70}")
    print("Rank | Cycle | Method        | Region           | Best Accuracy")
    print(f"{'-'*70}")
    
    for i, result in enumerate(results[:10]):
        region_str = f"{result['start']:4d}-{result['end']:4d}"
        print(f"{i+1:4d} | {result['cycle']:5d} | {result['method']:12} | {region_str} | {result['best_accuracy']:12.3f}")
    
    # Save best result
    if results:
        best_result = results[0]
        candidate_bits = best_result['candidate_bits']
        candidate_228 = candidate_bits[2:230]
        
        # Choose orientation
        if best_result['direct_accuracy'] >= best_result['inverted_accuracy']:
            final_bits = ''.join(str(bit) for bit in candidate_228)
            orientation = "direct"
        else:
            final_bits = ''.join('1' if bit == 0 else '0' for bit in candidate_228)
            orientation = "inverted"
        
        # Save results
        output_path = OUT_DIR / "best_candidate_mean_comparison.txt"
        with open(output_path, 'w') as f:
            f.write(final_bits + '\n')
        
        print(f"\n[+] BEST CANDIDATE FOUND:")
        print(f"    Cycle: {best_result['cycle']}")
        print(f"    Method: {best_result['method']}")
        print(f"    Region: {best_result['start']}-{best_result['end']}")
        print(f"    Orientation: {orientation}")
        print(f"    Accuracy: {best_result['best_accuracy']:.3f} ({best_result['best_accuracy']*100:.2f}%)")
        print(f"    Saved to: {output_path}")
        print(f"    First 64 bits: {final_bits[:64]}...")
        print(f"    Overlay plots: {plot_path}")
        
    else:
        print("\n[!] No valid results found")

if __name__ == "__main__":
    main()