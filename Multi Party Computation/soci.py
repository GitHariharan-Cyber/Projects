import random
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from scipy import stats
import gmpy2
import libnum
from paillier_NewOpt import KGen_NewOpt, Enc_NewOpt, Dec_NewOpt
import pre_compute
import sys
import csv
from datetime import datetime

WARMUP_TRIALS = 3

class DataProvider:
    def __init__(self, pk):
        self.data = random.randint(1,100)
        self.encryptedData = 0
        self.analystPK = pk
        self.public_key = pk  # Added missing attribute
    
    def setAnalystPK(self, pk):
        self.analystPK = pk

    def encryptData(self):
        if(self.analystPK == None):
            raise Exception("PK is not initialized")
        self.encryptedData = Enc_NewOpt(self.public_key, self.data)
    
    def getEncryptedData(self):
        return self.encryptedData

class S0:
    def __init__(self, dpList, N_Square):
        self.encryptedData = []  # Fixed: changed from self.data to self.encryptedData
        self.encryptedResult = 0
        for dp in dpList:
            self.encryptedData.append(dp.getEncryptedData())  # Fixed: using encryptedData

        self.N_square = N_Square
    
    def aggregate(self):
        total = 1  # Fixed: changed from 0 to 1 for multiplicative aggregation
        for x in self.encryptedData:
            total = gmpy2.mod(gmpy2.mul(total, x), self.N_square)
        self.encryptedResult = total

    def getEncryptedResult(self):
        return self.encryptedResult

class DataAnalyst:
    def __init__(self, use_precompute=True):
        self.private_key, self.public_key, _, _ = KGen_NewOpt()
        self.N = self.public_key['N']
        self.N_square = self.N ** 2
        self.use_precompute = use_precompute
        self.encryptedResult = 0
        
        if use_precompute:
            self.public_key['table'] = pre_compute.construct_table(
                self.public_key['h_N'], 
                self.N_square
            )

    def getPublicKey(self):
        return self.public_key

    def collectEncryptedResult(self, s0: S0):
        self.encryptedResult = s0.getEncryptedResult()  # Fixed: lowercase s0

    def decrypt(self) -> int:
        return Dec_NewOpt(self.private_key, self.encryptedResult)
    
    def getN_square(self):
        return self.N_square


def extract_stats(values):
    mean = np.mean(values)
    ci = stats.t.interval(0.95, len(values)-1, loc=mean, scale=stats.sem(values))
    return mean, (ci[1] - mean)

def run_benchmark(n_clients: int, n_iterations: int = 3) -> Tuple[float, float]:    
    da = DataAnalyst()
    dps = [DataProvider(da.getPublicKey()) for _ in range(n_clients)]  # Fixed: added parentheses
    s0 = S0(dps, da.getN_square())
    
    dp_times = []
    total_times = []
    for i in range(n_iterations):

        # run the data provider side, get the bottleneck time
        dp_time = 0
        for dp in dps:
            start = time.perf_counter()
            dp.encryptData()
            elapsed = time.perf_counter() - start
            if elapsed > dp_time:
                dp_time = elapsed
        dp_times.append(dp_time * 1000)  # Convert to milliseconds

        # run the server side
        start = time.perf_counter()
        s0.aggregate()
        s0_time = time.perf_counter() - start

        # run the data analyst side
        da.collectEncryptedResult(s0)
        start = time.perf_counter()
        result = da.decrypt()
        da_time = time.perf_counter() - start
        total_times.append((dp_time + s0_time + da_time) * 1000)  # Convert to milliseconds

    # Calculate communication cost per client
    dp_comm_cost = sys.getsizeof(dps[0].getEncryptedData())

    dp_times_mean, dp_times_ci = extract_stats(dp_times)
    total_times_mean, total_times_ci = extract_stats(total_times)
    
    return result, dp_times, total_times, dp_times_mean, dp_times_ci, total_times_mean, total_times_ci, dp_comm_cost, 0

def save_to_csv(client_counts: List[int], dp_all_times: List[List[float]], total_all_times: List[List[float]], 
                dp_means: List[float], dp_cis: List[float], total_means: List[float], total_cis: List[float]):
    """Save timing data to CSV files"""
    
    # Save DP times CSV
    with open('dp_times.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write header
        header = ['Data_Providers'] + [f'Run_{i+1}' for i in range(len(dp_all_times[0]))] + ['Mean', 'CI']
        writer.writerow(header)
        
        # Write data for each client count
        for i, n in enumerate(client_counts):
            row = [n] + dp_all_times[i] + [dp_means[i], dp_cis[i]]
            writer.writerow(row)
    
    # Save total times CSV
    with open('total_times.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write header
        header = ['Data_Providers'] + [f'Run_{i+1}' for i in range(len(total_all_times[0]))] + ['Mean', 'CI']
        writer.writerow(header)
        
        # Write data for each client count
        for i, n in enumerate(client_counts):
            row = [n] + total_all_times[i] + [total_means[i], total_cis[i]]
            writer.writerow(row)
    
    print("CSV files saved: dp_times.csv, total_times.csv")

def plot_results_soci_style(client_counts: List[int], means: List[float], cis: List[float], 
                           title: str, x_label: str, y_label: str, filename: str):
    """Plot results in SOCI style with k notation for data providers"""
    
    # Convert to numpy arrays
    x = np.array(client_counts)
    
    # Styling parameters
    colors = {
        'main': '#000000',   # Pure black
        'ci': '#7f7f7f',     # Gray for error bars
        'text': '#000000',   # Black text
        'grid': '#e0e0e0',   # Light gray grid
        'minor_grid': '#f0f0f0'  # Very light gray minor grid
    }
    
    plt.figure(figsize=(14, 7))
    plt.style.use('seaborn-whitegrid')
    
    # Prepare data
    y = np.array(means)
    y_err = np.array(cis)
    
    # Plot with levelers for CI
    plt.errorbar(
        x, y,
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
    plt.xlabel(x_label, fontsize=12, color=colors['text'])
    plt.ylabel(y_label, fontsize=12, color=colors['text'])
    plt.title(f'SOCI+: {title}', fontsize=14, color=colors['text'])
    
    # X-axis configuration (20k major, 5k minor)
    max_data_providers = 100500
    ax.set_xlim(0, max_data_providers)
    
    # Major ticks every 20k (labeled in k notation)
    major_x_ticks = np.arange(0, max_data_providers + 1, 20000)
    ax.set_xticks(major_x_ticks)
    ax.set_xticklabels([f"{int(x/1000)}k" for x in major_x_ticks], fontsize=10)
    
    # Minor ticks every 5k (unlabeled)
    minor_x_ticks = np.arange(0, max_data_providers + 1, 5000)
    ax.set_xticks(minor_x_ticks, minor=True)
    
    # Y-axis configuration (10ms major, 2ms minor)
    max_time = np.max(y + y_err) if len(y) > 0 else 50
    y_max = 10 * np.ceil(max_time / 10)  # Round up to nearest 10ms
    
    # Major ticks every 10ms (whole numbers)
    major_y_ticks = np.arange(0, y_max + 1, 10)
    ax.set_yticks(major_y_ticks)
    ax.set_yticklabels([f"{int(y)}" for y in major_y_ticks], fontsize=10)
    
    # Minor ticks every 2ms
    minor_y_ticks = np.arange(0, y_max + 1, 2)
    ax.set_yticks(minor_y_ticks, minor=True)
    ax.set_ylim(-1, y_max)
    
    # Grid configuration (all solid lines with specified thickness)
    ax.grid(which='major', linestyle='-', linewidth=0.8, color=colors['grid'])
    ax.grid(which='minor', linestyle='-', linewidth=0.4, color=colors['minor_grid'])
    
    # Remove top/right spines
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    
    plt.tight_layout()
    
    # Save plots
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f'{filename}_{timestamp}.svg', format='svg', dpi=1200, bbox_inches='tight')
    plt.savefig(f'{filename}_{timestamp}.pdf', format='pdf', dpi=1200, bbox_inches='tight')
    plt.savefig(f'{filename}_{timestamp}.png', format='png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    client_counts = [10, 100,250,750, 1000, 2500, 5000, 7500, 10000, 
                    20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000]
    n_iterations = 30
    result = 0
    
    print("Running benchmark...")
    dp_times_means, dp_times_cis = [], []
    total_times_means, total_times_cis = [], []
    dp_comms_means, dp_comms_cis = [], []
    
    # Store all individual timing data for CSV
    dp_all_times = []  # List of lists: each sublist contains all iterations for a client count
    total_all_times = []  # List of lists: each sublist contains all iterations for a client count
    
    for n in client_counts:
        # do warm-ups (excluded from measurements)
        for i in range(WARMUP_TRIALS):
            run_benchmark(n, 1)

        # Actual measurement trials
        result, dp_times, total_times, dp_times_mean, dp_times_ci, total_times_mean, total_times_ci, dp_comm_mean, dp_comm_ci = run_benchmark(n, n_iterations)
        
        # Store all individual timing data
        dp_all_times.append(dp_times)
        total_all_times.append(total_times)
        
        # Store statistics for plotting
        dp_times_means.append(dp_times_mean)
        dp_times_cis.append(dp_times_ci)
        total_times_means.append(total_times_mean)
        total_times_cis.append(total_times_ci)
        dp_comms_means.append(dp_comm_mean)
        dp_comms_cis.append(dp_comm_ci)
        
        print(f"Data Providers: {n:6d} | Time: {total_times_mean:.2f} ± {total_times_ci/2:.2f}ms | Result = {result}")
    
    # Save all timing data to CSV files
    save_to_csv(client_counts, dp_all_times, total_all_times, dp_times_means, dp_times_cis, total_times_means, total_times_cis)
    
    # Create plots in SOCI style
    plot_results_soci_style(client_counts, dp_times_means, dp_times_cis, 
                           'Data Provider Runtime', 'Number of Data Providers', 'Time [ms]', "soci_dp_times")
    plot_results_soci_style(client_counts, total_times_means, total_times_cis, 
                           'Total Runtime', 'Number of Data Providers', 'Time [ms]', "soci_total_times")

    print("\nBenchmark completed. Results saved to CSV files and plots")

if __name__ == "__main__":
    main()
