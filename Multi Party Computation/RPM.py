"""
RPM Protocol Benchmark - Addition Operation with Precise Grid Scaling and CSV Export
"""

import os
import time
import random
import numpy as np
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
from scipy import stats
from dataclasses import dataclass
import csv
import pandas as pd
from datetime import datetime

# System Parameters
NUM_SERVERS = 3
FIELD_SIZE = 2**61 - 1
MESSAGE_SIZE = 1024
SQUARE_NETWORK_LAYERS = 15
MPC_ROUND_TIME = 0.0001  # 0.1ms per round trip
CLIENT_PROCESSING_TIME = 0.00005  # 50μs per client operation
TRIALS = 10
CONFIDENCE = 0.95
WARMUP_RUNS = 3
MAX_MATRIX_SIZE = 10000

@dataclass
class DataProvider:
    id: int
    message: bytes
    value: int

class RPMBenchmark:
    def __init__(self):
        self.data_providers = []
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def simulate_mpc_round(self, rounds=1):
        """Precise MPC timing simulation (server-side time)"""
        if rounds > 100:
            time.sleep(rounds * MPC_ROUND_TIME)
        else:
            start = time.perf_counter()
            while time.perf_counter() - start < rounds * MPC_ROUND_TIME:
                pass
    
    def simulate_client_processing(self, operations=1):
        """Simulate client-side computation time"""
        if operations > 100:
            time.sleep(operations * CLIENT_PROCESSING_TIME)
        else:
            start = time.perf_counter()
            while time.perf_counter() - start < operations * CLIENT_PROCESSING_TIME:
                pass
    
    def _apply_addition(self, values: np.ndarray) -> np.ndarray:
        """Apply addition operation to values"""
        return np.cumsum(values) % FIELD_SIZE
    
    def variant1(self, k: int) -> Tuple[float, float, float, float]:
        """Variant 1 with timing breakdown (Matrix)"""
        # Warm-up
        for _ in range(WARMUP_RUNS):
            _ = np.random.permutation(min(k, 1000))
        
        # Client-side processing time
        client_start = time.perf_counter()
        self.simulate_client_processing(NUM_SERVERS)
        client_time = time.perf_counter() - client_start
        
        # Server-side processing time
        server_start = time.perf_counter()
        self.simulate_mpc_round(k)
        
        values = np.array([dp.value for dp in self.data_providers[:k]])
        result = self._apply_addition(values)
        
        server_time = time.perf_counter() - server_start
        total_time = client_time + server_time
        
        return total_time, client_time, server_time, result.mean()
    
    def variant3(self, k: int) -> Tuple[float, float, float, float]:
        """Variant 3 with timing breakdown (Square)"""
        # Warm-up
        block_size = int(np.ceil(np.sqrt(k)))
        for _ in range(WARMUP_RUNS):
            _ = np.random.permutation(min(block_size, 1000))
        
        # Client-side processing time
        client_start = time.perf_counter()
        self.simulate_client_processing(NUM_SERVERS)
        client_time = time.perf_counter() - client_start
        
        # Server-side processing time
        server_start = time.perf_counter()
        self.simulate_mpc_round(SQUARE_NETWORK_LAYERS * block_size)
        
        batch_size = min(10000, k)
        results = []
        for i in range(0, k, batch_size):
            batch = self.data_providers[i:i+batch_size]
            current = np.array([dp.value for dp in batch])
            results.extend(self._apply_addition(current))
        
        server_time = time.perf_counter() - server_start
        total_time = client_time + server_time
        
        return total_time, client_time, server_time, np.mean(results)
    
    def save_separate_variant_files(self, data_provider_counts: List[int], results: Dict):
        """Save separate CSV files for each variant with all timing data"""
        
        # Variant 1 file with all timing data
        self._save_complete_variant_file(data_provider_counts, results, 'V1', f"variant1_complete_{self.timestamp}.csv")
        
        # Variant 3 file with all timing data
        self._save_complete_variant_file(data_provider_counts, results, 'V3', f"variant3_complete_{self.timestamp}.csv")
        
        # Also save individual metric files
        self._save_variant_file(data_provider_counts, results, 'V1', 'total', f"variant1_total_{self.timestamp}.csv")
        self._save_variant_file(data_provider_counts, results, 'V1', 'client', f"variant1_client_{self.timestamp}.csv")
        self._save_variant_file(data_provider_counts, results, 'V1', 'server', f"variant1_server_{self.timestamp}.csv")
        
        self._save_variant_file(data_provider_counts, results, 'V3', 'total', f"variant3_total_{self.timestamp}.csv")
        self._save_variant_file(data_provider_counts, results, 'V3', 'client', f"variant3_client_{self.timestamp}.csv")
        self._save_variant_file(data_provider_counts, results, 'V3', 'server', f"variant3_server_{self.timestamp}.csv")
    
    def _save_complete_variant_file(self, data_provider_counts: List[int], results: Dict, variant: str, filename: str):
        """Save complete variant data with all timing metrics in one file"""
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                'Data_Providers', 
                f'{variant}_Total_Mean', f'{variant}_Total_CI',
                f'{variant}_Total_Time_Taken',
                f'{variant}_Client_Mean', f'{variant}_Client_CI', 
                f'{variant}_Client_Time_Taken',
                f'{variant}_Server_Mean', f'{variant}_Server_CI',
                f'{variant}_Server_Time_Taken'
            ])
            
            for i, k in enumerate(data_provider_counts):
                total_mean = results[f'{variant}_total']['mean'][i]
                total_ci = results[f'{variant}_total']['ci'][i]
                total_time = total_mean  # Time taken is the mean value
                
                client_mean = results[f'{variant}_client']['mean'][i]
                client_ci = results[f'{variant}_client']['ci'][i]
                client_time = client_mean  # Time taken is the mean value
                
                server_mean = results[f'{variant}_server']['mean'][i]
                server_ci = results[f'{variant}_server']['ci'][i]
                server_time = server_mean  # Time taken is the mean value
                
                writer.writerow([
                    k, 
                    total_mean, total_ci, total_time,
                    client_mean, client_ci, client_time,
                    server_mean, server_ci, server_time
                ])
    
    def _save_variant_file(self, data_provider_counts: List[int], results: Dict, variant: str, metric: str, filename: str):
        """Save individual variant metric to CSV file"""
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                'Data_Providers', 
                f'{variant}_{metric}_Mean', 
                f'{variant}_{metric}_CI',
                f'{variant}_{metric}_Time_Taken'
            ])
            
            for i, k in enumerate(data_provider_counts):
                key = f'{variant}_{metric}'
                mean = results[key]['mean'][i]
                ci = results[key]['ci'][i]
                time_taken = mean  # Time taken is the mean value
                writer.writerow([k, mean, ci, time_taken])
    
    def run_benchmark(self, data_provider_counts: List[int]) -> Dict[str, Dict]:
        """Execute benchmark for addition operation only"""
        results = {
            'V1_total': {'mean': [], 'ci': [], 'raw': []},
            'V1_client': {'mean': [], 'ci': [], 'raw': []},
            'V1_server': {'mean': [], 'ci': [], 'raw': []},
            'V3_total': {'mean': [], 'ci': [], 'raw': []},
            'V3_client': {'mean': [], 'ci': [], 'raw': []},
            'V3_server': {'mean': [], 'ci': [], 'raw': []}
        }
        
        for k in data_provider_counts:
            print(f"\nBenchmarking {k:,} data providers (Addition)...")
            self.data_providers = [DataProvider(i, os.urandom(MESSAGE_SIZE), random.randint(1, 100)) 
                                 for i in range(k)]
            
            v1_total_times, v1_client_times, v1_server_times = [], [], []
            v3_total_times, v3_client_times, v3_server_times = [], [], []
            
            for trial in range(1, TRIALS + 1):
                print(f"  Trial {trial}/{TRIALS}", end='\r', flush=True)
                
                # Variant 1 (Matrix)
                t1_total, t1_client, t1_server, _ = self.variant1(k)
                v1_total_times.append(t1_total)
                v1_client_times.append(t1_client)
                v1_server_times.append(t1_server)
                
                # Variant 3 (Square)
                t3_total, t3_client, t3_server, _ = self.variant3(k)
                v3_total_times.append(t3_total)
                v3_client_times.append(t3_client)
                v3_server_times.append(t3_server)
            
            # Process results
            for variant, total, client, server in [
                ('V1', v1_total_times, v1_client_times, v1_server_times),
                ('V3', v3_total_times, v3_client_times, v3_server_times)
            ]:
                for metric, times in [
                    ('total', total),
                    ('client', client),
                    ('server', server)
                ]:
                    key = f"{variant}_{metric}"
                    mean = np.mean(times)
                    ci = stats.t.interval(
                        CONFIDENCE, len(times)-1,
                        loc=mean, scale=stats.sem(times)
                    )[1] - mean
                    results[key]['mean'].append(mean)
                    results[key]['ci'].append(ci)
                    results[key]['raw'].append(times)
            
            print(f"Data Providers: {k:6,} | "
                  f"V1 Total: {results['V1_total']['mean'][-1]:.3f}s | "
                  f"V3 Total: {results['V3_total']['mean'][-1]:.3f}s")
        
        # Save separate files for each variant and metric
        self.save_separate_variant_files(data_provider_counts, results)
        
        return data_provider_counts, results

def create_performance_plots(data_provider_counts: List[int], results: Dict, timestamp: str):
    """Create three vectorized performance plots with exact grid scaling and CI levelers"""
    # Convert to numpy arrays
    x = np.array(data_provider_counts)
    
    # Styling parameters
    colors = {
        'main': '#000000',   # Pure black
        'ci': '#7f7f7f',     # Gray for error bars
        'text': '#000000',   # Black text
        'grid': '#e0e0e0',   # Light gray grid
        'minor_grid': '#f0f0f0'  # Very light gray minor grid
    }
    
    # Create plots for each category
    for category, title in [
        ('total', 'Total Execution Time'),
        ('client', 'Client-Side Processing Time'), 
        ('server', 'Server-Side Processing Time')
    ]:
        plt.figure(figsize=(14, 7))
        plt.style.use('seaborn-whitegrid')
        
        # Prepare data for both variants
        y_v1 = np.array(results[f'V1_{category}']['mean'])
        y_err_v1 = np.array(results[f'V1_{category}']['ci'])
        y_v3 = np.array(results[f'V3_{category}']['mean']) 
        y_err_v3 = np.array(results[f'V3_{category}']['ci'])
        
        # Plot both variants with levelers for CI
        plt.errorbar(
            x, y_v1,
            yerr=y_err_v1,
            fmt='o-',            # Circle markers with line
            color=colors['main'],
            ecolor=colors['ci'],
            label='Variant 1',
            capsize=4,           # Levelers for CI
            capthick=1.5,        # Thicker levelers
            markersize=5,
            linewidth=1.5,
            alpha=0.9
        )
        
        plt.errorbar(
            x, y_v3,
            yerr=y_err_v3,
            fmt='s--',           # Square markers with dashed line
            color=colors['main'],
            ecolor=colors['ci'],
            label='Variant 3',
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
        plt.title(f'RPM Protocol: {title} (Addition)', fontsize=14, color=colors['text'])
        
        # X-axis configuration (20k major, 5k minor)
        max_data_providers = 105001
        ax.set_xlim(0, max_data_providers)
        
        # Major ticks every 20k (labeled)
        major_x_ticks = np.arange(0, max_data_providers + 1, 20000)
        ax.set_xticks(major_x_ticks)
        ax.set_xticklabels([f"{int(x/1000)}k" for x in major_x_ticks], fontsize=10)
        
        # Minor ticks every 5k (unlabeled)
        minor_x_ticks = np.arange(0, max_data_providers + 1, 5000)
        ax.set_xticks(minor_x_ticks, minor=True)
        
        # Y-axis configuration (2s major, 0.5s minor)
        max_time = max(
            np.max(y_v1 + y_err_v1),
            np.max(y_v3 + y_err_v3)
        )
        y_max = 2 * np.ceil(max_time / 2)  # Round up to nearest even second
        
        # Major ticks every 2s (whole numbers)
        major_y_ticks = np.arange(0, y_max + 1, 2)
        ax.set_yticks(major_y_ticks)
        ax.set_yticklabels([f"{int(y)}" if y.is_integer() else f"{y:.0f}" for y in major_y_ticks], fontsize=10)
        
        # Minor ticks every 0.5s
        minor_y_ticks = np.arange(-0.5, y_max + 0.5, 0.5)
        ax.set_yticks(minor_y_ticks, minor=True)
        ax.set_ylim(-0.5, y_max)
        
        # Grid configuration (all solid lines)
        ax.grid(which='major', linestyle='-', linewidth=0.8, color=colors['grid'])
        ax.grid(which='minor', linestyle='-', linewidth=0.4, color=colors['minor_grid'])
        
        # Legend
        ax.legend(fontsize=10, loc='upper left', framealpha=1)
        
        # Remove top/right spines
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        
        plt.tight_layout()
        
        # Save plots with timestamp
        filename = f'rpm_addition_{category}_timing_{timestamp}.svg'.replace(' ', '_').lower()
        plt.savefig(filename, format='svg', dpi=1200, bbox_inches='tight')
        plt.savefig(filename.replace('.svg', '.pdf'), format='pdf', dpi=1200, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    print("=== RPM Protocol Benchmark (Addition Operation) ===")
    print(f"Configuration: {TRIALS} trials, {CONFIDENCE*100:.0f}% confidence")
    print(f"Timing parameters: {MPC_ROUND_TIME*1000:.1f}ms/round, {CLIENT_PROCESSING_TIME*1000:.1f}ms/client-op")
    print(f"Warm-up runs: {WARMUP_RUNS}")
    
    data_provider_counts = [10, 100, 250, 500, 750, 1000, 2500, 5000, 7500, 10000, 
                          20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000]
    
    benchmark = RPMBenchmark()
    counts, results = benchmark.run_benchmark(data_provider_counts)
    create_performance_plots(counts, results, benchmark.timestamp)
    
    print(f"\nGenerated 8 CSV files:")
    print(f"Complete files with all timing data:")
    print(f"- variant1_complete_{benchmark.timestamp}.csv")
    print(f"- variant3_complete_{benchmark.timestamp}.csv")
    print(f"\nIndividual metric files:")
    print(f"- variant1_total_{benchmark.timestamp}.csv")
    print(f"- variant1_client_{benchmark.timestamp}.csv")
    print(f"- variant1_server_{benchmark.timestamp}.csv")
    print(f"- variant3_total_{benchmark.timestamp}.csv")
    print(f"- variant3_client_{benchmark.timestamp}.csv")
    print(f"- variant3_server_{benchmark.timestamp}.csv")
    print(f"\nGenerated plot files:")
    print(f"- rpm_addition_total_timing_{benchmark.timestamp}.svg/.pdf")
    print(f"- rpm_addition_client_timing_{benchmark.timestamp}.svg/.pdf")
    print(f"- rpm_addition_server_timing_{benchmark.timestamp}.svg/.pdf")
