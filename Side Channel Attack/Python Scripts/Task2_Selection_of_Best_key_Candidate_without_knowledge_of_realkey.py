#!/usr/bin/env python3
"""
MeasuredTraceHamming_cyclic_best3.py
---------------------------------
- Compare measured trace key candidates against -5 and +14 offsets.
- For each candidate k, compute Hamming distance between:
  - candidate[k] and candidate[k-5] (direct)
  - candidate[k] and inverted(candidate[k+14])
- Find top 3 candidates with least combined Hamming distance.
"""

import numpy as np

# ---------------- CONFIG ----------------
INPUT_FILE = "candidate_bits.txt"   # measured power trace candidates
OFFSETS = [-5, +14]                           # offsets to check
# ----------------------------------------


def load_candidates_txt(filename):
    """Load candidates from text file (space-separated or concatenated bits)."""
    candidates = []
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if " " in line:  # space-separated
                bits = [int(x) for x in line.split()]
            else:            # continuous string
                bits = [int(x) for x in line]
            candidates.append(bits)
    return np.array(candidates)


def hamming_distance(c1, c2):
    """Compute Hamming distance between two bit arrays."""
    L = min(len(c1), len(c2))
    return np.sum(np.array(c1[:L]) != np.array(c2[:L]))


def find_best_candidates(candidates, offsets):
    """Find candidates with least combined Hamming distance for given offsets."""
    num = candidates.shape[0]
    results = []
    
    for k in range(num):
        # Calculate indices with cyclic wrapping
        idx_minus5 = (k + offsets[0]) % num
        idx_plus14 = (k + offsets[1]) % num
        
        # Get the candidates
        cand_k = candidates[k]
        cand_minus5 = candidates[idx_minus5]
        cand_plus14 = candidates[idx_plus14]
        
        # Invert the +14 candidate
        cand_plus14_inverted = [1 - bit for bit in cand_plus14]
        
        # Compute Hamming distances
        hamming_minus5 = hamming_distance(cand_k, cand_minus5)
        hamming_plus14_inverted = hamming_distance(cand_k, cand_plus14_inverted)
        
        # Total combined Hamming distance
        total_hamming = hamming_minus5 + hamming_plus14_inverted
        
        results.append({
            'index': k,
            'idx_minus5': idx_minus5,
            'idx_plus14': idx_plus14,
            'hamming_minus5': hamming_minus5,
            'hamming_plus14_inverted': hamming_plus14_inverted,
            'total_hamming': total_hamming
        })
    
    # Sort by total Hamming distance (ascending - lower is better)
    results.sort(key=lambda x: x['total_hamming'])
    
    return results


# ---------------- MAIN ----------------
if __name__ == "__main__":
    candidates = load_candidates_txt(INPUT_FILE)
    print(f"Loaded {candidates.shape[0]} candidates, each {candidates.shape[1]} bits long")
    
    results = find_best_candidates(candidates, OFFSETS)
    
    print(f"\nTop 3 candidates with least combined Hamming distance:")
    print("=" * 80)
    
    for i, result in enumerate(results[:3]):
        print(f"\n#{i+1}: Candidate index {result['index']}")
        print(f"    -5 offset: candidate[{result['index']}] vs candidate[{result['idx_minus5']}]")
        print(f"               Hamming distance: {result['hamming_minus5']}")
        print(f"    +14 offset: candidate[{result['index']}] vs INVERTED(candidate[{result['idx_plus14']}])")
        print(f"                Hamming distance: {result['hamming_plus14_inverted']}")
        print(f"    TOTAL Hamming distance: {result['total_hamming']}")
    
    # Print all results for reference
    print(f"\n{'='*80}")
    print("All candidates sorted by total Hamming distance:")
    print("Index\tTotal\t-5_dist\t+14_inv_dist\t-5_idx\t+14_idx")
    print("-" * 50)
    for result in results:
        print(f"{result['index']}\t{result['total_hamming']}\t{result['hamming_minus5']}\t{result['hamming_plus14_inverted']}\t\t{result['idx_minus5']}\t{result['idx_plus14']}")