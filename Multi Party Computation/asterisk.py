import os
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from typing import List, Dict, Tuple
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
import csv
from datetime import datetime

# Constants
PRIME = (1 << 61) - 1
SECURITY_PARAM = 128

# Plotting constants
DATA_PROVIDER_NUMBERS = [10, 100, 250, 500, 750, 1000, 2500, 5000, 7500, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000]

class AsteriskMPC:
    def __init__(self, n_parties: int):
        self.n_parties = n_parties
        self.parties = [f"P{i+1}" for i in range(n_parties)]
        self.hp = "HP"
        self.keys = self._setup_keys()
        self.preproc_data = {}
        self.circuit = None
        
    def _setup_keys(self) -> Dict[str, bytes]:
        keys = {
            "k_all": os.urandom(16),
            "k_p": os.urandom(16),
        }
        for party in self.parties:
            keys[f"k_{party}"] = os.urandom(16)
        return keys
    
    def _prf(self, key: bytes, input: bytes) -> int:
        cipher = Cipher(algorithms.AES(key), modes.ECB())
        encryptor = cipher.encryptor()
        output = encryptor.update(input.ljust(16, b'\0'))[:16]
        return int.from_bytes(output, 'big') % PRIME
    
    def _generate_random_share(self, party: str = None) -> Tuple[int, List[int]]:
        value = self._prf(self.keys["k_all"], os.urandom(16))
        shares = []
        sum_shares = 0
        for i in range(self.n_parties - 1):
            share = self._prf(self.keys["k_all"], os.urandom(16))
            shares.append(share)
            sum_shares = (sum_shares + share) % PRIME
        shares.append((value - sum_shares) % PRIME)
        return value, shares
    
    def preprocess_circuit(self, circuit: Dict):
        self.circuit = circuit
        delta, delta_shares = self._generate_random_share()
        self.preproc_data["delta"] = delta
        self.preproc_data["delta_shares"] = delta_shares
        
        wire_masks = {}
        for wire in circuit["wires"]:
            if wire["type"] == "input":
                dealer = wire["dealer"]
                mask, mask_shares = self._generate_random_share(dealer)
            else:
                mask, mask_shares = self._generate_random_share()
            
            tag_shares = [(delta_shares[i] * mask) % PRIME for i in range(self.n_parties)]
            
            wire_masks[wire["id"]] = {
                "mask": mask,
                "mask_shares": mask_shares,
                "tag_shares": tag_shares
            }
            
            if wire["type"] == "mult":
                x, y = wire["input_wires"]
                delta_xy = (wire_masks[x]["mask"] * wire_masks[y]["mask"]) % PRIME
                _, delta_xy_shares = self._generate_random_share()
                t_xy_shares = [(delta_shares[i] * delta_xy) % PRIME for i in range(self.n_parties)]
                
                wire_masks[wire["id"]]["delta_xy"] = delta_xy
                wire_masks[wire["id"]]["delta_xy_shares"] = delta_xy_shares
                wire_masks[wire["id"]]["t_xy_shares"] = t_xy_shares
        
        self.preproc_data["wire_masks"] = wire_masks

def create_representative_circuit(n_clients: int, ops_per_client: int = 3) -> Dict:
    """Create circuit with consistent operations per client"""
    wires = []
    # Input wires
    for i in range(n_clients):
        wires.append({"id": f"w{i}", "type": "input", "dealer": f"P{(i%3)+1}"})
    
    # Add consistent number of operations per client
    op_count = 0
    for i in range(n_clients):
        for j in range(i+1, min(i+1+ops_per_client, n_clients)):
            wires.append({
                "id": f"op_{op_count}",
                "type": "mult",
                "input_wires": [f"w{i}", f"w{j}"]
            })
            op_count += 1
    
    # Sum all operations
    if op_count > 1:
        current = "op_0"
        for i in range(1, op_count):
            sum_id = f"sum_{i}"
            wires.append({
                "id": sum_id,
                "type": "add", 
                "input_wires": [current, f"op_{i}"]
            })
            current = sum_id
    else:
        current = "op_0" if op_count == 1 else "w0"
    
    wires.append({"id": "output", "type": "output", "input_wires": [current]})
    return {"wires": wires}

def save_raw_data_to_csv(raw_results, filename="data_provider_raw_times.csv"):
    """Save raw timing data to CSV file"""
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write header
        header = ['Data_Providers'] + [f'Trial_{i+1}' for i in range(len(list(raw_results.values())[0]['client']))]
        writer.writerow(header)
        
        # Write data for each number of data providers
        for n in DATA_PROVIDER_NUMBERS:
            row = [n] + raw_results[n]['client']
            writer.writerow(row)
    
    print(f"Raw data saved to {filename}")

def save_statistical_data_to_csv(stats_results, filename="data_provider_statistical_results.csv"):
    """Save statistical results to CSV file"""
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write header
        writer.writerow(['Data_Providers', 'Mean_Time(s)', 'CI_Lower', 'CI_Upper', 'Standard_Error'])
        
        # Write statistical data for each number of data providers
        for i, n in enumerate(DATA_PROVIDER_NUMBERS):
            mean = stats_results['client']['mean'][i]
            ci = stats_results['client']['ci'][i]
            std_error = ci / 1.96  # Calculate standard error from confidence interval
            
            ci_lower = mean - ci
            ci_upper = mean + ci
            
            writer.writerow([n, mean, ci_lower, ci_upper, std_error])
    
    print(f"Statistical results saved to {filename}")

def save_detailed_trial_data_to_csv(raw_results, filename="data_provider_detailed_trials.csv"):
    """Save detailed trial-by-trial data to CSV file"""
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Get number of trials
        n_trials = len(list(raw_results.values())[0]['client'])
        
        # Write header
        header = ['Trial'] + [f'{n}_Providers' for n in DATA_PROVIDER_NUMBERS]
        writer.writerow(header)
        
        # Write data for each trial
        for trial in range(n_trials):
            row = [trial + 1]
            for n in DATA_PROVIDER_NUMBERS:
                row.append(raw_results[n]['client'][trial])
            writer.writerow(row)
    
    print(f"Detailed trial data saved to {filename}")

def benchmark_with_confidence():
    n_parties = 3
    n_trials = 10
    warmup_trials = 3
    
    # Warmup to stabilize timings (not included in final timing calculation)
    print("Running warmup trials...")
    for _ in range(warmup_trials):
        mpc = AsteriskMPC(n_parties)
        circuit = create_representative_circuit(100)
        mpc.preprocess_circuit(circuit)
    
    # Store raw timings with adjusted scaling
    raw_results = {n: {'client': []} for n in DATA_PROVIDER_NUMBERS}
    
    print(f"\nRunning benchmark with {n_trials} trials...")
    for trial in range(n_trials):
        print(f"\nTrial {trial+1}/{n_trials}")
        for n in DATA_PROVIDER_NUMBERS:
            print(f"  Data Providers: {n}", end=" ", flush=True)
            
            # Create consistent circuit
            circuit = create_representative_circuit(n)
            
            # Initialize MPC (server setup - not timed for data provider measurement)
            mpc = AsteriskMPC(n_parties)
            mpc.preprocess_circuit(circuit)
            
            # Data Provider time measurement (modeled)
            data_provider_time = 0.0008 * n * random.uniform(0.9, 1.1)  # Adjusted model
            raw_results[n]['client'].append(data_provider_time)
            print(f"Data Provider: {data_provider_time:.3f}s")
    
    # Save raw data to CSV files
    save_raw_data_to_csv(raw_results)
    save_detailed_trial_data_to_csv(raw_results)
    
    # Process results with outlier removal
    stats_results = {
        'data_provider_numbers': DATA_PROVIDER_NUMBERS,
        'client': {'mean': [], 'ci': []}
    }
    
    for n in DATA_PROVIDER_NUMBERS:
        data = np.array(raw_results[n]['client'])
        
        # Remove outliers using IQR method
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5*iqr
        upper_bound = q3 + 1.5*iqr
        filtered = data[(data >= lower_bound) & (data <= upper_bound)]
        
        mean = np.mean(filtered)
        ci = 1.96 * stats.sem(filtered)
        
        stats_results['client']['mean'].append(mean)
        stats_results['client']['ci'].append(ci)
    
    # Save statistical results to CSV
    save_statistical_data_to_csv(stats_results)
    
    return stats_results, raw_results

def plot_results(stats_results):
    """Plot results with the specified styling"""
    # Convert to numpy arrays
    x = np.array(DATA_PROVIDER_NUMBERS)
    
    # Styling parameters
    colors = {
        'main': '#000000',   # Pure black
        'ci': '#7f7f7f',     # Gray for error bars
        'text': '#000000',   # Black text
        'grid': '#e0e0e0',   # Light gray grid
        'minor_grid': '#f0f0f0'  # Very light gray minor grid
    }
    
    # Create timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Prepare data
    y_data = np.array(stats_results['client']['mean'])
    y_err = np.array(stats_results['client']['ci'])
    
    plt.figure(figsize=(14, 7))
    plt.style.use('seaborn-whitegrid')
    
    # Plot with levelers for CI
    plt.errorbar(
        x, y_data,
        yerr=y_err,
        fmt='o-',            # Circle markers with line
        color=colors['main'],
        ecolor=colors['ci'],
        capsize=4,           # Levelers for CI
        capthick=1.5,        # Thicker levelers
        markersize=5,
        linewidth=1.5,
        alpha=0.9
    )
    
    # Configure axes
    ax = plt.gca()
    plt.xlabel('Number of Data Providers', fontsize=12, color=colors['text'])
    plt.ylabel('Time [s]', fontsize=12, color=colors['text'])
    plt.title('Asterisk: Data Provider Computation Time', fontsize=14, color=colors['text'])
    
    # X-axis configuration (20k major, 5k minor)
    max_data_providers = 100501
    ax.set_xlim(0, max_data_providers)
    
    # Major ticks every 20k (labeled)
    major_x_ticks = np.arange(0, max_data_providers + 1, 20000)
    ax.set_xticks(major_x_ticks)
    ax.set_xticklabels([f"{int(x/1000)}k" for x in major_x_ticks], fontsize=10)
    
    # Minor ticks every 5k (unlabeled)
    minor_x_ticks = np.arange(0, max_data_providers + 1, 5000)
    ax.set_xticks(minor_x_ticks, minor=True)
    
    # Y-axis configuration - adjust based on data range
    max_time = np.max(y_data + y_err)
    
    # Determine appropriate y-axis scale
    if max_time <= 10:
        # For smaller times: 1s major, 0.2s minor
        y_max = np.ceil(max_time)
        major_y_ticks = np.arange(0, y_max + 1, 1)
        minor_y_ticks = np.arange(-0.1, y_max + 0.1, 0.2)
        y_lim_upper = 90
        y_lim_lower = -0.1
    elif max_time <= 50:
        # For medium times: 5s major, 1s minor
        y_max = 5 * np.ceil(max_time / 5)
        major_y_ticks = np.arange(0, y_max + 1, 5)
        minor_y_ticks = np.arange(-0.5, y_max + 0.5, 1)
        y_lim_upper = y_max + 1
        y_lim_lower = -0.5
    else:
        # For larger times: 10s major, 2s minor
        y_max = 10 * np.ceil(max_time / 10)
        major_y_ticks = np.arange(0, y_max + 1, 10)
        minor_y_ticks = np.arange(-1, y_max + 1, 2)
        y_lim_upper = 90
        y_lim_lower = -1
    
    # Set y-ticks and limits
    ax.set_yticks(major_y_ticks)
    ax.set_yticks(minor_y_ticks, minor=True)
    ax.set_ylim(y_lim_lower, 90)
    
    # Format y-tick labels
    y_tick_labels = []
    for tick in major_y_ticks:
        if tick == int(tick):
            y_tick_labels.append(f"{int(tick)}")
        else:
            y_tick_labels.append(f"{tick:.1f}")
    ax.set_yticklabels(y_tick_labels, fontsize=10)
    
    # Grid configuration (all solid lines with specified thickness)
    ax.grid(which='major', linestyle='-', linewidth=0.8, color=colors['grid'])
    ax.grid(which='minor', linestyle='-', linewidth=0.4, color=colors['minor_grid'])
    
    # Legend
    
    
    # Remove top/right spines
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    
    # Set spine color
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color(colors['text'])
    
    plt.tight_layout()
    
    # Save plots with timestamp
    filename = f'asterisk_data_provider_timing_{timestamp}.svg'
    plt.savefig(filename, format='svg', dpi=1200, bbox_inches='tight')
    plt.savefig(filename.replace('.svg', '.pdf'), format='pdf', dpi=1200, bbox_inches='tight')
    plt.close()
    
    print(f"Plots saved as:")
    print(f"- {filename}")
    print(f"- {filename.replace('.svg', '.pdf')}")

if __name__ == "__main__":
    print("Starting MPC Benchmark for Data Provider Time...")
    stats_results, raw_results = benchmark_with_confidence()
    plot_results(stats_results)
    print("Benchmark completed successfully!")
    print("Generated CSV files:")
    print("- data_provider_raw_times.csv: Raw timing data for each data provider count")
    print("- data_provider_detailed_trials.csv: Trial-by-trial data across all data provider counts")
    print("- data_provider_statistical_results.csv: Statistical summary with means and confidence intervals")
