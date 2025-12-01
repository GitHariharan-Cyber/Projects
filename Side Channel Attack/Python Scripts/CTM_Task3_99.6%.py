#!/usr/bin/env python3
"""
Simple Mean Comparison Attack - No Known Key Version with Visualization
- Uses separation quality metrics instead of known key validation
- Automatically picks best candidate based on signal quality
- Plots overlay of slots for the best cycle
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

# ---------------- CONFIGURATION ----------------
CSV_PATH = Path("C1--pt--00003.csv")
COL_IDX = 1
START_IDX = 2627250
NUM_SLOTS = 230
CYCLES_PER_SLOT = 54
SAMPLES_PER_CYCLE = 1250
SLOT_LEN = CYCLES_PER_SLOT * SAMPLES_PER_CYCLE

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

def evaluate_separation_quality(slot_means, candidate_bits):
    """
    Evaluate how well the two groups (0s and 1s) are separated.
    Returns a quality score (higher = better separation)
    """
    if len(np.unique(candidate_bits)) < 2:
        return 0.0  # Only one group - bad separation
    
    group0_means = slot_means[candidate_bits == 0]
    group1_means = slot_means[candidate_bits == 1]
    
    if len(group0_means) == 0 or len(group1_means) == 0:
        return 0.0
    
    # Metric 1: Distance between means (normalized by overall std)
    mean0, mean1 = np.mean(group0_means), np.mean(group1_means)
    std_overall = np.std(slot_means)
    distance_score = abs(mean1 - mean0) / (std_overall + 1e-8)
    
    # Metric 2: Low variance within groups, high variance between groups
    var_within = (np.var(group0_means) + np.var(group1_means)) / 2
    var_between = np.var([mean0, mean1])
    separation_score = var_between / (var_within + 1e-8)
    
    # Metric 3: T-test statistic (how statistically different are the groups)
    t_stat = abs(stats.ttest_ind(group0_means, group1_means).statistic)
    
    # Combined quality score
    quality_score = distance_score * separation_score * (1 + t_stat/10)
    
    return quality_score

def mean_comparison_attack(cycle_data, use_full_cycle=True, region_start=0, region_end=None):
    """Simple mean comparison attack."""
    if use_full_cycle:
        region_data = cycle_data
    else:
        if region_end is None:
            region_end = cycle_data.shape[1]
        region_data = cycle_data[:, region_start:region_end]
    
    slot_means = np.mean(region_data, axis=1)
    threshold = np.median(slot_means)
    candidate_bits = (slot_means > threshold).astype(int)
    
    return candidate_bits, slot_means, threshold

def plot_best_cycle_analysis(slot_data, best_result, output_dir):
    """Create comprehensive plots for the best cycle."""
    
    best_cycle = best_result['cycle']
    best_method = best_result['method']
    best_start = best_result['start']
    best_end = best_result['end']
    candidate_bits = best_result['candidate_bits']
    slot_means = best_result['slot_means']
    
    cycle_data = slot_data[:, best_cycle, :]
    x_axis = np.arange(SAMPLES_PER_CYCLE)
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(16, 12))
    
    # Plot 1: All slots overlay (black)
    ax1 = plt.subplot(2, 2, 1)
    for slot_idx in range(NUM_SLOTS):
        ax1.plot(x_axis, cycle_data[slot_idx], color='black', alpha=0.1, linewidth=0.3)
    
    # Highlight the best region
    ax1.axvspan(best_start, best_end, color='red', alpha=0.3, 
                label=f'Analysis Region: {best_start}-{best_end}')
    ax1.set_title(f'Cycle {best_cycle} - All Slots Overlay\n({best_method})', 
                  fontsize=12, fontweight='bold')
    ax1.set_xlabel('Sample Index')
    ax1.set_ylabel('Amplitude')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Slots colored by extracted bits
    ax2 = plt.subplot(2, 2, 2)
    colors = ['blue' if bit == 0 else 'red' for bit in candidate_bits]
    for slot_idx in range(NUM_SLOTS):
        ax2.plot(x_axis, cycle_data[slot_idx], color=colors[slot_idx], alpha=0.15, linewidth=0.3)
    
    ax2.axvspan(best_start, best_end, color='green', alpha=0.3, 
                label=f'Analysis Region: {best_start}-{best_end}')
    ax2.set_title(f'Cycle {best_cycle} - Colored by Extracted Bits\nBlue=0, Red=1', 
                  fontsize=12, fontweight='bold')
    ax2.set_xlabel('Sample Index')
    ax2.set_ylabel('Amplitude')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Mean traces for each group
    ax3 = plt.subplot(2, 2, 3)
    group0_indices = np.where(candidate_bits == 0)[0]
    group1_indices = np.where(candidate_bits == 1)[0]
    
    if len(group0_indices) > 0:
        mean_trace_0 = np.mean(cycle_data[group0_indices], axis=0)
        ax3.plot(x_axis, mean_trace_0, color='blue', linewidth=2, 
                label=f'Group 0 (bits=0), n={len(group0_indices)}')
    
    if len(group1_indices) > 0:
        mean_trace_1 = np.mean(cycle_data[group1_indices], axis=0)
        ax3.plot(x_axis, mean_trace_1, color='red', linewidth=2, 
                label=f'Group 1 (bits=1), n={len(group1_indices)}')
    
    ax3.axvspan(best_start, best_end, color='orange', alpha=0.3, 
                label=f'Analysis Region: {best_start}-{best_end}')
    ax3.set_title(f'Cycle {best_cycle} - Mean Traces for Each Group', 
                  fontsize=12, fontweight='bold')
    ax3.set_xlabel('Sample Index')
    ax3.set_ylabel('Amplitude')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Histogram of slot means showing separation
    ax4 = plt.subplot(2, 2, 4)
    
    group0_means = slot_means[candidate_bits == 0]
    group1_means = slot_means[candidate_bits == 1]
    
    if len(group0_means) > 0:
        ax4.hist(group0_means, bins=30, alpha=0.7, color='blue', 
                label=f'Group 0 (n={len(group0_means)})')
    
    if len(group1_means) > 0:
        ax4.hist(group1_means, bins=30, alpha=0.7, color='red', 
                label=f'Group 1 (n={len(group1_means)})')
    
    threshold = np.median(slot_means)
    ax4.axvline(threshold, color='black', linestyle='--', linewidth=2, 
               label=f'Threshold: {threshold:.4f}')
    
    ax4.set_title(f'Distribution of Slot Means\nQuality Score: {best_result["quality_score"]:.3f}', 
                  fontsize=12, fontweight='bold')
    ax4.set_xlabel('Mean Amplitude in Analysis Region')
    ax4.set_ylabel('Frequency')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Add overall title
    plt.suptitle(f'Best Cycle Analysis: Cycle {best_cycle} ({best_method})\n'
                f'Region: {best_start}-{best_end}, Mean Difference: {best_result["mean_difference"]:.4f}', 
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    # Save the figure
    plot_path = output_dir / f"best_cycle_{best_cycle}_{best_method}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[+] Best cycle analysis plot saved to: {plot_path}")
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
    
    print("Testing cycles with mean comparison (no known key)...")
    results = []
    
    # Test each cycle with different approaches
    for cycle in range(CYCLES_PER_SLOT):
        cycle_data = slot_data[:, cycle, :]
        
        methods = [
            ('full_cycle', True, 0, None),
            ('first_half', False, 0, SAMPLES_PER_CYCLE//2),
            ('second_half', False, SAMPLES_PER_CYCLE//2, SAMPLES_PER_CYCLE),
            ('middle_section', False, SAMPLES_PER_CYCLE//4, 3*SAMPLES_PER_CYCLE//4)
        ]
        
        for method_name, use_full, start, end in methods:
            candidate_bits, slot_means, threshold = mean_comparison_attack(
                cycle_data, use_full_cycle=use_full, region_start=start, region_end=end
            )
            
            # Evaluate separation quality instead of known key accuracy
            quality_score = evaluate_separation_quality(slot_means, candidate_bits)
            
            # Also calculate basic statistics about the separation
            group0_means = slot_means[candidate_bits == 0]
            group1_means = slot_means[candidate_bits == 1]
            
            mean_diff = abs(np.mean(group1_means) - np.mean(group0_means)) if len(group0_means) > 0 and len(group1_means) > 0 else 0
            
            results.append({
                'cycle': cycle,
                'method': method_name,
                'start': start,
                'end': end if end is not None else SAMPLES_PER_CYCLE,
                'quality_score': quality_score,
                'mean_difference': mean_diff,
                'group0_size': len(group0_means),
                'group1_size': len(group1_means),
                'candidate_bits': candidate_bits,
                'slot_means': slot_means,
                'threshold': threshold
            })
    
    # Sort by quality score (higher = better separation)
    results.sort(key=lambda x: x['quality_score'], reverse=True)
    
    print(f"\n{'='*80}")
    print("TOP 10 RESULTS (Based on Separation Quality)")
    print(f"{'='*80}")
    print("Rank | Cycle | Method        | Region           | Quality Score | Mean Diff | Group Sizes")
    print(f"{'-'*80}")
    
    for i, result in enumerate(results[:10]):
        region_str = f"{result['start']:4d}-{result['end']:4d}"
        group_sizes = f"{result['group0_size']:2d}/{result['group1_size']:2d}"
        print(f"{i+1:4d} | {result['cycle']:5d} | {result['method']:12} | {region_str} | "
              f"{result['quality_score']:12.3f} | {result['mean_difference']:8.3f} | {group_sizes}")
    
    # Save best result and create plots
    if results:
        best_result = results[0]
        candidate_bits = best_result['candidate_bits']
        
        # For the final output, skip first 2 slots to get 228 bits
        candidate_228 = candidate_bits[2:230]
        final_bits = ''.join(str(bit) for bit in candidate_228)
        
        # Save results
        output_path = OUT_DIR / "best_candidate_unknown_key.txt"
        with open(output_path, 'w') as f:
            f.write(final_bits + '\n')
        
        # Also save the "inverted" version since we don't know orientation
        inverted_bits = ''.join('1' if bit == 0 else '0' for bit in candidate_228)
        inverted_path = OUT_DIR / "best_candidate_inverted.txt"
        with open(inverted_path, 'w') as f:
            f.write(inverted_bits + '\n')
        
        # Create visualization for the best cycle
        plot_path = plot_best_cycle_analysis(slot_data, best_result, OUT_DIR)
        
        print(f"\n[+] BEST CANDIDATE FOUND (Unknown Key):")
        print(f"    Cycle: {best_result['cycle']}")
        print(f"    Method: {best_result['method']}")
        print(f"    Region: {best_result['start']}-{best_result['end']}")
        print(f"    Quality Score: {best_result['quality_score']:.3f}")
        print(f"    Mean Difference: {best_result['mean_difference']:.4f}")
        print(f"    Group Sizes: {best_result['group0_size']} zeros, {best_result['group1_size']} ones")
        print(f"    Saved to: {output_path} (direct)")
        print(f"    Saved to: {inverted_path} (inverted)")
        print(f"    Visualization: {plot_path}")
        print(f"    First 64 bits (direct): {final_bits[:64]}...")
        print(f"    First 64 bits (inverted): {inverted_bits[:64]}...")
        
        # Show why this was chosen
        print(f"\n[+] Why this was chosen:")
        print(f"    - Clear separation between groups (quality score: {best_result['quality_score']:.3f})")
        print(f"    - Large mean difference: {best_result['mean_difference']:.4f}")
        print(f"    - Balanced groups: {best_result['group0_size']}/{best_result['group1_size']}")
        
    else:
        print("\n[!] No valid results found")

if __name__ == "__main__":
    main()