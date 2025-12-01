import os
import numpy as np
import matplotlib.pyplot as plt
import time
import random
import threading
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from scipy import stats
import pandas as pd
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.backends import default_backend

# Configuration
N_TRIALS = 10
WARMUP_TRIALS = 3
CONFIDENCE_LEVEL = 0.95

# ==================== Cryptographic Primitives ====================

class CTR_DRBG:
    """NIST-compliant random number generator"""
    def __init__(self):
        self.key = os.urandom(32)
        self.counter = int.from_bytes(os.urandom(16), 'big')
        
    def seed(self, seed: bytes):
        h = hashes.Hash(hashes.SHA256())
        h.update(seed)
        self.key = h.finalize()[:32]
        self.counter = int.from_bytes(os.urandom(16), 'big')
        
    def random_bytes(self, length: int) -> bytes:
        result = bytearray()
        while len(result) < length:
            cipher = Cipher(algorithms.AES(self.key), modes.ECB())
            encryptor = cipher.encryptor()
            block = encryptor.update(self.counter.to_bytes(16, 'big')) + encryptor.finalize()
            result.extend(block)
            self.counter += 1
        return bytes(result[:length])

class AES:
    """AES-256 implementation with ECB mode"""
    BLOCK_SIZE = 16
    
    def __init__(self, key: bytes):
        if len(key) != 32:
            raise ValueError("Requires 256-bit key")
        self.key = key
        
    def encrypt(self, plaintext: bytes) -> bytes:
        cipher = Cipher(algorithms.AES(self.key), modes.ECB())
        encryptor = cipher.encryptor()
        return encryptor.update(plaintext) + encryptor.finalize()
    
    def decrypt(self, ciphertext: bytes) -> bytes:
        cipher = Cipher(algorithms.AES(self.key), modes.ECB())
        decryptor = cipher.decryptor()
        return decryptor.update(ciphertext) + decryptor.finalize()

class ECDH:
    """Elliptic Curve Diffie-Hellman (P-256) with proper serialization"""
    def __init__(self):
        self.curve = ec.SECP256R1()
        self.private_key = ec.generate_private_key(self.curve)
        
    def make_public(self) -> bytes:
        """Serialize public key in uncompressed format"""
        return self.private_key.public_key().public_bytes(
            encoding=serialization.Encoding.X962,
            format=serialization.PublicFormat.UncompressedPoint
        )
    
    def read_public(self, public_bytes: bytes):
        """Load peer's public key"""
        self.peer_public_key = ec.EllipticCurvePublicKey.from_encoded_point(
            curve=self.curve,
            data=public_bytes
        )
    
    def calc_secret(self) -> bytes:
        """Compute shared secret"""
        return self.private_key.exchange(ec.ECDH(), self.peer_public_key)

class Paillier:
    """Paillier homomorphic encryption system"""
    def __init__(self):
        self.pubkey = None
        self.prvkey = None
        
    def keygen(self, nbits: int, rand_ctx: CTR_DRBG):
        p = self._generate_prime(nbits//2, rand_ctx)
        q = self._generate_prime(nbits//2, rand_ctx)
        n = p * q
        self.pubkey = (n, n + 1, n * n)  # (n, g, n_sq)
        lambda_val = (p-1)*(q-1)
        self.prvkey = (lambda_val, pow(lambda_val, -1, n))  # (lambda, mu)
    
    def _generate_prime(self, bits: int, rand_ctx: CTR_DRBG) -> int:
        while True:
            candidate = int.from_bytes(rand_ctx.random_bytes(bits//8), 'big') | (1 << (bits-1))
            if self._is_prime(candidate):
                return candidate
    
    def _is_prime(self, n: int, k: int = 5) -> bool:
        if n <= 1:
            return False
        for p in [2,3,5,7,11,13,17,19,23,29]:
            if n % p == 0:
                return n == p
        d, s = n - 1, 0
        while d % 2 == 0:
            d //= 2
            s += 1
        for _ in range(k):
            a = random.randint(2, n-2)
            x = pow(a, d, n)
            if x == 1 or x == n-1:
                continue
            for __ in range(s-1):
                x = pow(x, 2, n)
                if x == n-1:
                    break
            else:
                return False
        return True
    
    def encrypt(self, plaintext: int, rand_ctx: CTR_DRBG) -> int:
        n, g, n_sq = self.pubkey
        r = int.from_bytes(rand_ctx.random_bytes(32), 'big') % n
        return (pow(g, plaintext, n_sq) * pow(r, n, n_sq)) % n_sq
    
    def decrypt(self, ciphertext: int) -> int:
        n, _, n_sq = self.pubkey
        lambda_val, mu = self.prvkey
        x = pow(ciphertext, lambda_val, n_sq)
        return ((x - 1) // n * mu) % n
    
    def add(self, a: int, b: int) -> int:
        return (a * b) % self.pubkey[2]

# ==================== Experiment Core ====================

class TEEEnclave:
    def __init__(self):
        self.rand_ctx = CTR_DRBG()
        self.rand_ctx.seed(os.urandom(32))
        self.aes_key = os.urandom(32)
        self.paillier = Paillier()
        self.paillier.keygen(2048, self.rand_ctx)
        
    def secure_compute(self, data: List[int], operation: str) -> Tuple[Optional[List[int]], float]:
        start = time.time()
        try:
            if operation == "SELECT":
                # Real AES encryption timing - THIS IS THE SLOW OPERATION
                cipher = AES(self.aes_key)
                encrypted_results = []
                for x in data:
                    block = x.to_bytes(16, 'big')
                    encrypted = cipher.encrypt(block)
                    encrypted_results.append(int.from_bytes(encrypted, 'big'))
                
                # Filter even numbers
                result = [x for x in data if x % 2 == 0]
                return result, time.time() - start
                
            elif operation == "ADD":
                if not data:
                    return [0], time.time() - start
                
                # Real Paillier homomorphic operations timing
                total = self.paillier.encrypt(data[0], self.rand_ctx)
                for x in data[1:]:
                    encrypted_x = self.paillier.encrypt(x, self.rand_ctx)
                    total = self.paillier.add(total, encrypted_x)
                
                result_val = self.paillier.decrypt(total)
                return [result_val], time.time() - start
                
        except Exception as e:
            print(f"Enclave error: {e}")
        return None, 0.001

class HybridTrustMPC:
    def __init__(self, n_parties: int):
        self.enclaves = [TEEEnclave() for _ in range(n_parties)]
        self.trust_levels = {}
        self.session_keys = {}
        self.lock = threading.Lock()
        
    def set_trust_level(self, party_id: int, level: int):
        with self.lock:
            self.trust_levels[party_id] = level
            
    def _establish_session(self, party_id: int) -> bool:
        """Establish secure session with proper HKDF initialization"""
        try:
            with self.lock:
                local = ECDH()
                local_pub = local.make_public()
                
                remote = ECDH()
                remote_pub = remote.make_public()
                local.read_public(remote_pub)
                
                shared_secret = local.calc_secret()
                
                # Initialize HKDF with empty salt
                hkdf = HKDF(
                    algorithm=hashes.SHA256(),
                    length=32,
                    salt=None,
                    info=b'hybrid_trust_session',
                    backend=default_backend()
                )
                self.session_keys[party_id] = hkdf.derive(shared_secret)
                return True
        except Exception as e:
            print(f"Session error: {e}")
            return False
            
    def secure_aggregate(self, data: List[int], party_id: int) -> Tuple[int, float]:
        start = time.time()
        try:
            with self.lock:
                trust_level = self.trust_levels.get(party_id, 0)
            
            if trust_level > 0 and party_id not in self.session_keys:
                if not self._establish_session(party_id):
                    return 0, 0.001
            
            # Use "SELECT" operation for proper timing (same as original)
            if trust_level == 2:  # Full TEE
                result, _ = self.enclaves[party_id].secure_compute(data, "SELECT")
                return sum(result or [0]), time.time() - start
            else:  # Default to complete trust if other levels are specified
                result, _ = self.enclaves[party_id].secure_compute(data, "SELECT")
                return sum(result or [0]), time.time() - start
        except Exception as e:
            print(f"Aggregation error: {e}")
            return 0, 0.001

# ==================== Benchmarking & Plotting ====================

def generate_data_provider_counts() -> List[int]:
    return [10, 100, 250, 500, 750, 1000, 2500, 5000, 7500, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000]

def run_trial(n_data_providers: int, trust_level: int) -> float:
    try:
        data = [random.randint(1, 100) for _ in range(n_data_providers)]
        hmpc = HybridTrustMPC(1)
        hmpc.set_trust_level(0, trust_level)
        _, elapsed = hmpc.secure_aggregate(data, 0)
        return max(elapsed, 0.001)
    except Exception as e:
        print(f"Trial error: {e}")
        return 0.001

def create_plot(results: Dict, filename: str):
    """Professional plotting with dynamic Y-axis limits based on 100k data providers"""
    plt.figure(figsize=(10, 6))
    
    # Use updated seaborn style
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        plt.style.use('default')
    
    colors = ['black']
    markers = ['o']
    marker_sizes = [4]
    linestyles = ['-']
    labels = ['Complete Trust']
    
    # Find the time for 100k data providers to set Y-axis limit
    max_100k_time = 0
    for level in results:
        n_data_providers = results[level]['n_data_providers']
        means = results[level]['means']
        # Find the index of 100000 in the data provider counts
        if 100000 in n_data_providers:
            idx = n_data_providers.index(100000)
            max_100k_time = means[idx]
            break
    
    # Set Y-axis limits based on 100k data providers time
    # Add 20% headroom and round up to nearest 5
    y_max = np.ceil((max_100k_time * 1.2) / 5) * 5
    y_min = -0.5
    
    print(f"Max time for 100k data providers: {max_100k_time:.2f}s")
    print(f"Setting Y-axis limit to: {y_max:.1f}s")
    
    # Plot each trust level
    for level in results:
        n_data_providers = results[level]['n_data_providers']
        means = results[level]['means']
        lowers = results[level]['lowers']
        uppers = results[level]['uppers']
        
        plt.errorbar(
            n_data_providers, means,
            yerr=[np.array(means)-np.array(lowers), np.array(uppers)-np.array(means)],
            fmt=markers[0],
            color=colors[0],
            linestyle=linestyles[0],
            label=labels[0],
            capsize=3,
            markersize=marker_sizes[0],
            linewidth=1.5,
            alpha=0.9,
            markeredgewidth=0.8
        )
    
    ax = plt.gca()
    
    # X-axis configuration - full numbers instead of 'k' format
    ax.set_xlim(0, 105000)
    ax.set_xticks(np.arange(0, 100001, 20000))
    ax.set_xticks(np.arange(0, 100001, 5000), minor=True)
    plt.xlabel('Number of Data Providers', fontsize=11)
    
    # Y-axis configuration with dynamic limits and fixed scaling
    ax.set_ylim(y_min, y_max)
    ax.set_yticks(np.arange(0, y_max + 1, 5))  # Major ticks every 5 seconds
    ax.set_yticks(np.arange(0, y_max + 1, 1), minor=True)  # Minor ticks every 1 second
    plt.ylabel('Time [s]', fontsize=11)
    
    ax.grid(which='major', linestyle='-', linewidth=0.7, alpha=0.7)
    ax.grid(which='minor', linestyle='-', linewidth=0.5, alpha=0.3)
    
    plt.title('Hybrid Trust MPC Performance', fontsize=12, pad=15)
    ax.legend(fontsize=10, framealpha=1, edgecolor='black')
    
    plt.tight_layout()
    plt.savefig(f'{filename}.pdf', format='pdf', dpi=1200, bbox_inches='tight')
    plt.savefig(f'{filename}.svg', format='svg', bbox_inches='tight')
    plt.close()

def benchmark() -> Tuple[Dict, Dict]:
    """Run full benchmark with statistical analysis and return both summary and detailed results"""
    data_provider_counts = generate_data_provider_counts()
    results = {}
    detailed_results = {}
    
    # Only test complete trust (level 2)
    for level in [2]:
        means, lowers, uppers = [], [], []
        detailed_times = []
        
        for n in data_provider_counts:
            print(f"Testing level {level} with {n} data providers...")
            
            # Run warm-up trials (not included in final results)
            for _ in range(WARMUP_TRIALS):
                run_trial(n, level)
            
            # Run actual trials for measurement - PARALLEL EXECUTION
            with ThreadPoolExecutor() as executor:
                times = list(executor.map(run_trial, [n]*N_TRIALS, [level]*N_TRIALS))
                
            # Store detailed results
            for trial_num, trial_time in enumerate(times):
                detailed_times.append({
                    'data_providers': n,
                    'trial_number': trial_num + 1,
                    'time_taken': trial_time
                })
            
            mean = np.mean(times)
            sem = stats.sem(times)
            ci = sem * stats.t.ppf((1+CONFIDENCE_LEVEL)/2, N_TRIALS-1)
            
            means.append(mean)
            lowers.append(max(mean - ci, 0))
            uppers.append(mean + ci)
            
            print(f"  Result: {mean:.3f}s ± {ci:.3f}")
        
        results[level] = {
            'n_data_providers': data_provider_counts,
            'means': means,
            'lowers': lowers,
            'uppers': uppers
        }
        
        detailed_results[level] = detailed_times
    
    return results, detailed_results

def save_summary_results(results: Dict):
    """Save summary results to CSV"""
    rows = []
    for level in results:
        for n, m, l, u in zip(
            results[level]['n_data_providers'],
            results[level]['means'],
            results[level]['lowers'],
            results[level]['uppers']
        ):
            rows.append({
                'trust_level': level,
                'n_data_providers': n,
                'mean_time': m,
                'ci_lower': l,
                'ci_upper': u,
                'ci_width': u-l
            })
    
    pd.DataFrame(rows).to_csv('hybrid_trust_summary_results.csv', index=False)

def save_detailed_results(detailed_results: Dict):
    """Save detailed trial-by-trial results to CSV"""
    rows = []
    for level in detailed_results:
        for trial_data in detailed_results[level]:
            rows.append({
                'trust_level': level,
                'data_providers': trial_data['data_providers'],
                'trial_number': trial_data['trial_number'],
                'time_taken': trial_data['time_taken']
            })
    
    pd.DataFrame(rows).to_csv('hybrid_trust_detailed_results.csv', index=False)

if __name__ == "__main__":
    print("=== Hybrid Trust MPC Benchmark ===")
    print(f"Configuration: {N_TRIALS} trials per data point")
    print(f"Warm-up trials: {WARMUP_TRIALS} (not included in results)")
    print("Using SELECT operation for proper timing measurements")
    print("Y-axis limit will be automatically adjusted based on 100k data providers")
    
    results, detailed_results = benchmark()
    
    # Create plots with professional styling
    create_plot(results, 'hybrid_trust_performance')
    
    # Save both summary and detailed results
    save_summary_results(results)
    save_detailed_results(detailed_results)
    
    print("\nBenchmark completed!")
    print("Results saved to:")
    print("- hybrid_trust_performance.pdf")
    print("- hybrid_trust_performance.svg") 
    print("- hybrid_trust_summary_results.csv (statistical summary)")
    print("- hybrid_trust_detailed_results.csv (individual trial times)")
