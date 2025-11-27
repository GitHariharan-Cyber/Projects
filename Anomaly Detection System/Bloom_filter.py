import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from scipy import stats
import mmh3
from bitarray import bitarray
import re

class ICSBloomFilterAnalyzer:
    """
    ICS Anomaly Detection System - Bloom Filter Only
    Based on: "Multi-level Anomaly Detection in Industrial Control Systems"
    """
    
    def __init__(self):
        self.discretizer = PaperDiscretizer()
        self.bloom_filter = None
        self.unique_signatures = None
        
    def run_bloom_filter_pipeline(self, arff_file_path):
        """Run pipeline: Discretization → Bloom Filter Training"""
        print("="*60)
        print("ICS ANOMALY DETECTION - BLOOM FILTER ONLY")
        print("="*60)
        
        # Step 1: Data Loading and Discretization
        print("\n1. DATA LOADING AND DISCRETIZATION")
        print("-" * 40)
        
        data = self.discretizer.load_arff_data(arff_file_path)
        self.discretizer.fit(data, use_clean_data=False)
        signatures, _, self.unique_signatures = self.discretizer.discretize_dataset(data)
        
        # Step 2: Bloom Filter Training
        print("\n2. BLOOM FILTER TRAINING")
        print("-" * 40)
        
        self.bloom_filter = self._train_bloom_filter(signatures)
        
        # Step 3: System Test
        print("\n3. SYSTEM TEST")
        print("-" * 40)
        
        self._test_bloom_filter(signatures)
        
        print("\n" + "="*60)
        print("BLOOM FILTER SYSTEM TRAINED SUCCESSFULLY!")
        print("="*60)
        
        return self.bloom_filter
    
    def _train_bloom_filter(self, signatures, false_positive_rate=0.03):
        """Train Bloom Filter with paper-compliant parameters"""
        print("Training Bloom Filter...")
        
        # Use only unique signatures for training
        unique_sigs = list(set(signatures))
        capacity = len(unique_sigs)
        
        # Initialize Bloom Filter
        bloom_filter = PaperBloomFilter(capacity, false_positive_rate)
        
        # Add unique signatures
        bloom_filter.add_batch(unique_sigs)
        
        # Print statistics
        stats = bloom_filter.get_statistics()
        print("Bloom Filter Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        return bloom_filter
    
    def _test_bloom_filter(self, signatures):
        """Test Bloom Filter detection"""
        print("Testing Bloom Filter detection...")
        
        if len(signatures) < 10:
            print("Not enough data for testing")
            return
            
        # Test with sample signatures
        test_signatures = signatures[:10]
        
        print("Sample detection results:")
        for i, signature in enumerate(test_signatures):
            is_anomaly, confidence = self.bloom_filter.package_level_detection(signature)
            status = "ANOMALY" if is_anomaly else "NORMAL"
            print(f"  {i+1}. {status}: {signature[:50]}...")
    
    def detect_anomaly(self, packet_data):
        """Anomaly detection using Bloom Filter only"""
        signature = self.discretizer.generate_signature(packet_data)
        is_anomaly, confidence = self.bloom_filter.package_level_detection(signature)
        
        if is_anomaly:
            return True, f"Package-level anomaly: Unknown signature (confidence: {confidence:.3f})"
        else:
            return False, f"Normal packet (confidence: {confidence:.3f})"


class PaperDiscretizer:
    """Paper-compliant feature discretization"""
    
    def __init__(self):
        self.boundaries = {}
        self.pid_clusters = None
        self.feature_statistics = {}
    
    def load_arff_data(self, file_path):
        """Load ARFF format dataset"""
        data = []
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        in_data_section = False
        for line in lines:
            line = line.strip()
            if line.lower() == '@data':
                in_data_section = True
                continue
            if in_data_section and line and not line.startswith('@'):
                row = [x.strip().strip("'") for x in line.split(',')]
                data.append(row)
        
        print(f"Loaded {len(data)} samples")
        return data
    
    def analyze_feature_statistics(self, data):
        """Analyze paper-specified features only"""
        features = {
            'setpoint': [], 'pressure': [], 'crc_rate': [], 'time_intervals': [], 'pid_params': []
        }
        
        # Extract timestamps for time intervals
        timestamps = []
        for row in data:
            if len(row) > 16 and row[16] != '?':
                try:
                    timestamps.append(float(row[16]))
                except ValueError:
                    continue
        
        # Calculate time intervals
        for i in range(1, len(timestamps)):
            features['time_intervals'].append(timestamps[i] - timestamps[i-1])
        
        # Extract paper-specified features
        for row in data:
            # Setpoint (index 3)
            if len(row) > 3 and row[3] != '?':
                try:
                    features['setpoint'].append(float(row[3]))
                except ValueError:
                    continue
            
            # Pressure (index 13)
            if len(row) > 13 and row[13] != '?':
                try:
                    features['pressure'].append(float(row[13]))
                except ValueError:
                    continue
            
            # CRC rate (index 14)
            if len(row) > 14 and row[14] != '?':
                try:
                    features['crc_rate'].append(float(row[14]))
                except ValueError:
                    continue
            
            # PID parameters (indices 4,5,6,7,8)
            pid_params = []
            valid_pid = True
            for idx in [4, 5, 6, 7, 8]:
                if len(row) <= idx or row[idx] == '?':
                    valid_pid = False
                    break
                try:
                    pid_params.append(float(row[idx]))
                except ValueError:
                    valid_pid = False
                    break
            
            if valid_pid and len(pid_params) == 5:
                features['pid_params'].append(pid_params)
        
        # Analyze statistics
        self.feature_statistics = {}
        for feature_name, values in features.items():
            if values and feature_name != 'pid_params':
                values = np.array(values)
                stats_dict = self._calculate_robust_statistics(values, feature_name)
                self.feature_statistics[feature_name] = stats_dict
        
        return self.feature_statistics
    
    def _calculate_robust_statistics(self, values, feature_name):
        """Calculate statistics with outlier detection"""
        if len(values) == 0:
            return {
                'all_values': values,
                'count': 0,
                'mean': 0,
                'std': 0,
                'min': 0,
                'max': 0,
                'normal_min': 0,
                'normal_max': 0,
            }
        
        stats_dict = {
            'all_values': values,
            'count': len(values),
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
        }
        
        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers = values[(values < lower_bound) | (values > upper_bound)]
        normal_values = values[(values >= lower_bound) & (values <= upper_bound)]
        
        stats_dict.update({
            'outliers_count': len(outliers),
            'normal_min': np.min(normal_values) if len(normal_values) > 0 else stats_dict['min'],
            'normal_max': np.max(normal_values) if len(normal_values) > 0 else stats_dict['max'],
        })
        
        print(f"{feature_name:15} | Range: {stats_dict['normal_min']:6.2f}-{stats_dict['normal_max']:6.2f} | "
              f"Outliers: {stats_dict['outliers_count']:3d}")
        
        return stats_dict
    
    def fit(self, data, use_clean_data=True):
        """Calculate discretization boundaries"""
        print("\nCalculating discretization boundaries...")
        
        self.analyze_feature_statistics(data)
        
        # Pressure - 20 even intervals + outlier
        if 'pressure' in self.feature_statistics and len(self.feature_statistics['pressure']['all_values']) > 0:
            stats = self.feature_statistics['pressure']
            if use_clean_data and len(stats['normal_values']) > 0:
                self.boundaries['pressure'] = {'low': stats['normal_min'], 'high': stats['normal_max']}
            else:
                self.boundaries['pressure'] = {'low': stats['min'], 'high': stats['max']}
        
        # Setpoint - 10 even intervals + outlier
        if 'setpoint' in self.feature_statistics and len(self.feature_statistics['setpoint']['all_values']) > 0:
            stats = self.feature_statistics['setpoint']
            if use_clean_data and len(stats['normal_values']) > 0:
                self.boundaries['setpoint'] = {'low': stats['normal_min'], 'high': stats['normal_max']}
            else:
                self.boundaries['setpoint'] = {'low': stats['min'], 'high': stats['max']}
        
        # Time intervals - K-means 2 clusters + outlier
        if 'time_intervals' in self.feature_statistics and len(self.feature_statistics['time_intervals']['all_values']) > 0:
            stats = self.feature_statistics['time_intervals']
            time_data = stats['normal_values'].reshape(-1, 1) if use_clean_data and len(stats['normal_values']) > 0 else stats['all_values'].reshape(-1, 1)
            
            if len(time_data) >= 2:
                kmeans_time = KMeans(n_clusters=2, random_state=42)
                kmeans_time.fit(time_data)
                centers = sorted(kmeans_time.cluster_centers_.flatten())
                self.boundaries['time_interval'] = {'low': centers[0], 'high': centers[1]}
        
        # CRC rate - K-means 2 clusters + outlier
        if 'crc_rate' in self.feature_statistics and len(self.feature_statistics['crc_rate']['all_values']) > 0:
            stats = self.feature_statistics['crc_rate']
            crc_data = stats['normal_values'].reshape(-1, 1) if use_clean_data and len(stats['normal_values']) > 0 else stats['all_values'].reshape(-1, 1)
            
            if len(crc_data) >= 2:
                kmeans_crc = KMeans(n_clusters=2, random_state=42)
                kmeans_crc.fit(crc_data)
                centers = sorted(kmeans_crc.cluster_centers_.flatten())
                self.boundaries['crc_rate'] = {'low': centers[0], 'high': centers[1]}
        
        # PID parameters - K-means clustering
        pid_params = self._extract_pid_parameters(data)
        if len(pid_params) >= 32:
            self.pid_clusters = KMeans(n_clusters=32, random_state=42)
            self.pid_clusters.fit(pid_params)
            print(f"PID parameters clustered into 32 groups")
        
        return self
    
    def _extract_pid_parameters(self, data):
        """Extract PID parameters for clustering"""
        pid_params = []
        for row in data:
            pid_array = []
            valid_pid = True
            for idx in [4, 5, 6, 7, 8]:
                if len(row) <= idx or row[idx] == '?':
                    valid_pid = False
                    break
                try:
                    pid_array.append(float(row[idx]))
                except ValueError:
                    valid_pid = False
                    break
            
            if valid_pid and len(pid_array) == 5:
                pid_params.append(pid_array)
        
        return np.array(pid_params)
    
    def discretize_pressure(self, value):
        """Discretize pressure into 20 categories + outliers"""
        if 'pressure' not in self.boundaries:
            return 'pressure_unknown'
        
        bounds = self.boundaries['pressure']
        
        if value < bounds['low']:
            return 'pressure_low_outlier'
        elif value > bounds['high']:
            return 'pressure_high_outlier'
        
        interval_width = (bounds['high'] - bounds['low']) / 20
        for i in range(20):
            lower_bound = bounds['low'] + i * interval_width
            upper_bound = bounds['low'] + (i + 1) * interval_width
            
            if lower_bound <= value < upper_bound:
                return f'pressure_{i+1:02d}'
        
        return 'pressure_20'
    
    def discretize_setpoint(self, value):
        """Discretize setpoint into 10 categories + outliers"""
        if 'setpoint' not in self.boundaries:
            return 'setpoint_unknown'
        
        bounds = self.boundaries['setpoint']
        
        if value < bounds['low']:
            return 'setpoint_low_outlier'
        elif value > bounds['high']:
            return 'setpoint_high_outlier'
        
        interval_width = (bounds['high'] - bounds['low']) / 10
        for i in range(10):
            lower_bound = bounds['low'] + i * interval_width
            upper_bound = bounds['low'] + (i + 1) * interval_width
            
            if lower_bound <= value < upper_bound:
                return f'setpoint_{i+1:02d}'
        
        return 'setpoint_10'
    
    def discretize_time_interval(self, value):
        """Discretize time interval using K-means clusters"""
        if 'time_interval' not in self.boundaries:
            return 'time_unknown'
        
        bounds = self.boundaries['time_interval']
        if value < bounds['low']:
            return 'time_short'
        elif value < bounds['high']:
            return 'time_long'
        else:
            return 'time_outlier'
    
    def discretize_crc_rate(self, value):
        """Discretize CRC rate using K-means clusters"""
        if 'crc_rate' not in self.boundaries:
            return 'crc_unknown'
        
        bounds = self.boundaries['crc_rate']
        if value < bounds['low']:
            return 'crc_low'
        elif value < bounds['high']:
            return 'crc_high'
        else:
            return 'crc_outlier'
    
    def discretize_pid_params(self, pid_array):
        """Discretize PID parameters using K-means clusters"""
        if self.pid_clusters is None:
            return 'pid_unknown'
        
        pid_array = np.array(pid_array).reshape(1, -1)
        cluster_idx = self.pid_clusters.predict(pid_array)[0]
        return f'pid_cluster_{cluster_idx+1:02d}'
    
    def generate_signature(self, row, prev_timestamp=None):
        """Generate signature for Bloom Filter"""
        signature_parts = []
        
        # Add all features with paper-compliant discretization
        features_to_add = [
            (0, 'addr', lambda x: f"addr_{x}"),
            (1, 'func', lambda x: f"func_{x}"),
            (2, 'len', lambda x: f"len_{x}"),
            (3, 'setpoint', lambda x: self.discretize_setpoint(float(x)) if x != '?' else 'setpoint_unknown'),
            (9, 'mode', lambda x: f"mode_{x}"),
            (10, 'scheme', lambda x: f"scheme_{x}"),
            (11, 'pump', lambda x: f"pump_{x}"),
            (12, 'solenoid', lambda x: f"solenoid_{x}"),
            (13, 'pressure', lambda x: self.discretize_pressure(float(x)) if x != '?' else 'pressure_unknown'),
            (14, 'crc', lambda x: self.discretize_crc_rate(float(x)) if x != '?' else 'crc_unknown'),
            (15, 'cmd', lambda x: f"cmd_{x}"),
        ]
        
        for idx, prefix, func in features_to_add:
            if len(row) > idx and row[idx] != '?':
                try:
                    signature_parts.append(func(row[idx]))
                except (ValueError, TypeError):
                    signature_parts.append(f"{prefix}_unknown")
            else:
                signature_parts.append(f"{prefix}_unknown")
        
        # Add PID parameters
        pid_params = []
        valid_pid = True
        for idx in [4, 5, 6, 7, 8]:
            if len(row) <= idx or row[idx] == '?':
                valid_pid = False
                break
            try:
                pid_params.append(float(row[idx]))
            except (ValueError, TypeError):
                valid_pid = False
                break
        
        if valid_pid:
            signature_parts.append(self.discretize_pid_params(pid_params))
        else:
            signature_parts.append("pid_unknown")
        
        # Add time interval
        if prev_timestamp is not None and len(row) > 16 and row[16] != '?':
            try:
                current_time = float(row[16])
                time_interval = current_time - prev_timestamp
                signature_parts.append(self.discretize_time_interval(time_interval))
            except (ValueError, TypeError):
                signature_parts.append("time_unknown")
        else:
            signature_parts.append("time_unknown")
        
        return "|".join(signature_parts)
    
    def create_feature_vector(self, row):
        """Create numerical feature vector (simplified for Bloom filter only)"""
        # For Bloom filter only, we don't need feature vectors
        return np.array([])
    
    def discretize_dataset(self, data):
        """Discretize entire dataset and generate signatures"""
        print("Generating signatures...")
        
        signatures = []
        feature_vectors = []  # Empty for Bloom filter only
        prev_timestamp = None
        
        for i, row in enumerate(data):
            if i % 10000 == 0:
                print(f"  Processed {i}/{len(data)} packets...")
            
            signature = self.generate_signature(row, prev_timestamp)
            signatures.append(signature)
            feature_vectors.append(self.create_feature_vector(row))
            
            if len(row) > 16 and row[16] != '?':
                try:
                    prev_timestamp = float(row[16])
                except (ValueError, TypeError):
                    pass
        
        unique_signatures = set(signatures)
        print(f"Generated {len(signatures)} signatures")
        print(f"Unique signatures: {len(unique_signatures)}")
        
        return signatures, np.array(feature_vectors), unique_signatures


class PaperBloomFilter:
    """Paper-compliant Bloom Filter implementation"""
    
    def __init__(self, capacity, false_positive_rate=0.03):
        self.capacity = capacity
        self.false_positive_rate = false_positive_rate
        
        self.size = self._calculate_size(capacity, false_positive_rate)
        self.hash_functions = self._calculate_hash_functions(self.size, capacity)
        
        self.bit_array = bitarray(self.size)
        self.bit_array.setall(0)
    
    def _calculate_size(self, n, p):
        return int(-(n * np.log(p)) / (np.log(2) ** 2))
    
    def _calculate_hash_functions(self, m, n):
        k = int((m / n) * np.log(2))
        return max(1, k)
    
    def _hashes(self, signature):
        hashes = []
        for i in range(self.hash_functions):
            hash_val = mmh3.hash(signature, i) % self.size
            hashes.append(hash_val)
        return hashes
    
    def add(self, signature):
        hashes = self._hashes(signature)
        for hash_val in hashes:
            self.bit_array[hash_val] = 1
    
    def contains(self, signature):
        hashes = self._hashes(signature)
        for hash_val in hashes:
            if not self.bit_array[hash_val]:
                return False
        return True
    
    def add_batch(self, signatures):
        print(f"Adding {len(signatures)} unique signatures...")
        for signature in signatures:
            self.add(signature)
    
    def package_level_detection(self, packet_signature):
        if self.contains(packet_signature):
            return False, 1.0  # Normal
        else:
            return True, 0.0   # Anomaly
    
    def get_statistics(self):
        bits_set = self.bit_array.count()
        fill_ratio = bits_set / self.size
        actual_fpp = (1 - np.exp(-self.hash_functions * self.capacity / self.size)) ** self.hash_functions
        
        return {
            'capacity': self.capacity,
            'size_bits': self.size,
            'hash_functions': self.hash_functions,
            'bits_set': bits_set,
            'fill_ratio': fill_ratio,
            'theoretical_fpp': self.false_positive_rate,
            'actual_fpp': actual_fpp,
            'memory_usage_kb': self.size / 8192
        }


# Main execution
if __name__ == "__main__":
    # Initialize the Bloom filter system
    ics_analyzer = ICSBloomFilterAnalyzer()
    
    # Run Bloom filter pipeline only
    arff_file = "Dataset.arff"  # Replace with your dataset path
    try:
        bloom_filter = ics_analyzer.run_bloom_filter_pipeline(arff_file)
        
        print("\nBloom Filter system ready for anomaly detection!")
        print("Use ics_analyzer.detect_anomaly(packet_data) for real-time detection")
        
    except FileNotFoundError:
        print(f"Error: Dataset file '{arff_file}' not found.")
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()