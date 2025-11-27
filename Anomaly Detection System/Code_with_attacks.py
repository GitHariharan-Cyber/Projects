import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import mmh3
from bitarray import bitarray
import pickle
import re

class PaperCompliantICSAnalyzer:
    """
    Complete ICS Anomaly Detection System as per:
    "Multi-level Anomaly Detection in Industrial Control Systems via Package Signatures and LSTM networks"
    Cheng Feng, Tingting Li and Deeph Chana - Imperial College London
    """
    
    def __init__(self):
        self.discretizer = PaperDiscretizer()
        self.bloom_filter = None
        self.lstm_detector = None
        self.unique_signatures = None
        
    def run_complete_pipeline(self, arff_file_path):
        """Run complete pipeline: Data Separation → Discretization → Bloom Filter → LSTM Training"""
        print("="*80)
        print("PAPER-COMPLIANT ICS ANOMALY DETECTION SYSTEM")
        print("="*80)
        
        # Step 1: Data Loading and Separation
        print("\n1. DATA LOADING AND SEPARATION")
        print("-" * 40)
        
        data = self.discretizer.load_arff_data(arff_file_path)
        
        # CRITICAL: Separate normal and attack data using labels
        normal_data, attack_data = self._separate_normal_attack_data(data)
        
        print(f"Normal packets: {len(normal_data)}")
        print(f"Attack packets: {len(attack_data)}")
        
        # Step 2: Discretization on NORMAL DATA ONLY
        print("\n2. DISCRETIZATION (NORMAL DATA ONLY)")
        print("-" * 40)
        
        self.discretizer.fit(normal_data, use_clean_data=False)
        normal_signatures, normal_feature_vectors, self.unique_signatures = self.discretizer.discretize_dataset(normal_data)
        
        # Step 3: Bloom Filter Training on NORMAL SIGNATURES ONLY
        print("\n3. BLOOM FILTER TRAINING (NORMAL SIGNATURES ONLY)")
        print("-" * 40)
        
        self.bloom_filter = self._train_bloom_filter(normal_signatures)
        
        # Step 4: LSTM Model Training on NORMAL SEQUENCES ONLY
        print("\n4. LSTM NETWORK TRAINING (NORMAL SEQUENCES ONLY)")
        print("-" * 40)
        
        self.lstm_detector = self._train_lstm_model(normal_feature_vectors, normal_signatures, self.unique_signatures)
        
        # Step 5: System Evaluation on ATTACK DATA
        print("\n5. SYSTEM EVALUATION ON ATTACK DATA")
        print("-" * 40)
        
        if len(attack_data) > 0:
            self._evaluate_on_attack_data(attack_data)
        
        print("\n" + "="*80)
        print("COMPLETE SYSTEM TRAINED AND EVALUATED SUCCESSFULLY!")
        print("="*80)
        
        return self.bloom_filter, self.lstm_detector
    
    def _separate_normal_attack_data(self, data):
        """Separate data into normal and attack using binary_result labels"""
        normal_data = []
        attack_data = []
        
        for row in data:
            # Assuming binary_result is at index 17 (adjust based on your ARFF format)
            if len(row) > 17 and row[17] == '0':  # Normal
                normal_data.append(row)
            elif len(row) > 17 and row[17] == '1':  # Attack
                attack_data.append(row)
            else:
                # Handle cases where label is missing or unknown
                normal_data.append(row)  # Default to normal if label missing
        
        return normal_data, attack_data
    
    def _train_bloom_filter(self, signatures, false_positive_rate=0.03):
        """Train Bloom Filter with paper-compliant parameters on NORMAL signatures only"""
        print("Training Bloom Filter on NORMAL signatures only...")
        
        # Use only unique signatures for training (as per paper)
        unique_sigs = list(set(signatures))
        capacity = len(unique_sigs)
        
        # Initialize Bloom Filter
        bloom_filter = PaperBloomFilter(capacity, false_positive_rate)
        
        # Add unique NORMAL signatures only
        bloom_filter.add_batch(unique_sigs)
        
        # Print statistics
        stats = bloom_filter.get_statistics()
        print("Bloom Filter Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        return bloom_filter
    
    def _train_lstm_model(self, feature_vectors, signatures, unique_signatures):
        """Train LSTM model with paper-compliant architecture on NORMAL data only"""
        print("Training LSTM Network on NORMAL sequences only...")
        
        # Initialize LSTM detector
        lstm_detector = PaperLSTMDetector(
            sequence_length=10,
            input_features=feature_vectors.shape[1],
            num_signatures=len(unique_signatures)
        )
        
        # Build model architecture (exactly as per paper)
        lstm_detector.build_model()
        
        # Create signature mapping for softmax output
        signature_mapping = lstm_detector.create_signature_mapping(unique_signatures)
        
        # Create sequences for training (10-packet windows) from NORMAL data only
        X_sequences, y_sequences = lstm_detector.create_sequences(
            feature_vectors, signatures, signature_mapping
        )
        
        if len(X_sequences) == 0:
            print("Error: No sequences created. Check your data.")
            return None
        
        # PAPER-COMPLIANT SPLIT: 60% train, 20% validation, 20% test
        print("Splitting data: 60% train, 20% validation, 20% test...")
        
        # First split: 60% train, 40% temp
        X_train, X_temp, y_train, y_temp = train_test_split(
            X_sequences, y_sequences, test_size=0.4, random_state=42, shuffle=False
        )
        
        # Second split: 50% of temp (20% of original) for validation, 50% (20% of original) for test
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, shuffle=False
        )
        
        print(f"Training samples: {X_train.shape[0]} ({X_train.shape[0]/len(X_sequences)*100:.1f}%)")
        print(f"Validation samples: {X_val.shape[0]} ({X_val.shape[0]/len(X_sequences)*100:.1f}%)")
        print(f"Test samples: {X_test.shape[0]} ({X_test.shape[0]/len(X_sequences)*100:.1f}%)")
        
        # Verify the split percentages
        total = len(X_sequences)
        print(f"Verification - Train: {X_train.shape[0]/total*100:.1f}%, "
              f"Val: {X_val.shape[0]/total*100:.1f}%, "
              f"Test: {X_test.shape[0]/total*100:.1f}%")
        
        # Train with paper parameters: 50 epochs, batch_size=32
        print("\nStarting LSTM training (50 epochs)...")
        history = lstm_detector.train(X_train, y_train, X_val, y_val, epochs=50, batch_size=32)
        
        # Evaluate on test set
        test_accuracy = lstm_detector.evaluate(X_test, y_test)
        print(f"Test Accuracy on NORMAL data: {test_accuracy:.4f}")
        
        return lstm_detector
    
    def _evaluate_on_attack_data(self, attack_data):
        """Evaluate the system on attack data"""
        print("Evaluating system on attack data...")
        
        # Process attack data
        attack_signatures, attack_feature_vectors, _ = self.discretizer.discretize_dataset(attack_data)
        
        # Test Bloom Filter on attacks
        bloom_detections = 0
        for signature in attack_signatures:
            is_anomaly, _ = self.bloom_filter.package_level_detection(signature)
            if is_anomaly:
                bloom_detections += 1
        
        bloom_detection_rate = bloom_detections / len(attack_signatures) if attack_signatures else 0
        print(f"Bloom Filter Attack Detection Rate: {bloom_detection_rate:.4f}")
        
        # Test LSTM on attack sequences (if we have enough data)
        if len(attack_feature_vectors) >= 11:
            successful_attack_detections = 0
            total_attack_tests = 0
            
            # Test multiple attack sequences
            for i in range(min(100, len(attack_feature_vectors) - 10)):
                test_sequence = attack_feature_vectors[i:i+10]
                actual_next_signature = attack_signatures[i+10]
                
                lstm_anomaly, _, _ = self.lstm_detector.time_series_anomaly_detection(
                    test_sequence, actual_next_signature
                )
                
                if lstm_anomaly:
                    successful_attack_detections += 1
                total_attack_tests += 1
            
            if total_attack_tests > 0:
                lstm_attack_detection_rate = successful_attack_detections / total_attack_tests
                print(f"LSTM Attack Detection Rate: {lstm_attack_detection_rate:.4f}")
                
                # Combined detection rate
                combined_detection_rate = max(bloom_detection_rate, lstm_attack_detection_rate)
                print(f"Combined System Attack Detection Rate: {combined_detection_rate:.4f}")
    
    def detect_anomaly(self, packet_data, previous_packets=None):
        """Complete anomaly detection for a new packet"""
        # Generate signature
        signature = self.discretizer.generate_signature(packet_data, previous_packets)
        
        # Layer 1: Bloom Filter check
        bloom_anomaly, _ = self.bloom_filter.package_level_detection(signature)
        if bloom_anomaly:
            return True, "Package-level anomaly: Unknown signature"
        
        # Layer 2: LSTM check (if we have sequence context)
        if previous_packets and len(previous_packets) >= 10:
            feature_vectors = [self.discretizer.create_feature_vector(p) for p in previous_packets[-10:]]
            input_sequence = np.array(feature_vectors)
            
            lstm_anomaly, confidence, _ = self.lstm_detector.time_series_anomaly_detection(
                input_sequence, signature
            )
            
            if lstm_anomaly:
                return True, f"Time-series anomaly: Low confidence ({confidence:.3f})"
        
        return False, "Normal packet"


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
                'q1': 0,
                'median': 0,
                'q3': 0,
                'outliers_count': 0,
                'outliers_percentage': 0,
                'normal_min': 0,
                'normal_max': 0,
                'normal_values': values
            }
        
        stats_dict = {
            'all_values': values,
            'count': len(values),
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'q1': np.percentile(values, 25),
            'median': np.percentile(values, 50),
            'q3': np.percentile(values, 75)
        }
        
        iqr = stats_dict['q3'] - stats_dict['q1']
        lower_bound = stats_dict['q1'] - 1.5 * iqr
        upper_bound = stats_dict['q3'] + 1.5 * iqr
        
        outliers = values[(values < lower_bound) | (values > upper_bound)]
        normal_values = values[(values >= lower_bound) & (values <= upper_bound)]
        
        stats_dict.update({
            'outliers_count': len(outliers),
            'outliers_percentage': len(outliers) / len(values) * 100 if len(values) > 0 else 0,
            'normal_min': np.min(normal_values) if len(normal_values) > 0 else stats_dict['min'],
            'normal_max': np.max(normal_values) if len(normal_values) > 0 else stats_dict['max'],
            'normal_values': normal_values
        })
        
        print(f"{feature_name:15} | Range: {stats_dict['normal_min']:6.2f}-{stats_dict['normal_max']:6.2f} | "
              f"Outliers: {stats_dict['outliers_count']:3d}")
        
        return stats_dict
    
    def fit(self, data, use_clean_data=True):
        """Calculate discretization boundaries for paper-specified features"""
        print("\nCalculating discretization boundaries...")
        
        self.analyze_feature_statistics(data)
        
        # Pressure - 20 even intervals + outlier (Paper Table III)
        if 'pressure' in self.feature_statistics and len(self.feature_statistics['pressure']['all_values']) > 0:
            stats = self.feature_statistics['pressure']
            if use_clean_data and len(stats['normal_values']) > 0:
                self.boundaries['pressure'] = {'low': stats['normal_min'], 'high': stats['normal_max']}
            else:
                self.boundaries['pressure'] = {'low': stats['min'], 'high': stats['max']}
            print(f"Pressure boundaries: {self.boundaries['pressure']}")
        
        # Setpoint - 10 even intervals + outlier (Paper Table III)
        if 'setpoint' in self.feature_statistics and len(self.feature_statistics['setpoint']['all_values']) > 0:
            stats = self.feature_statistics['setpoint']
            if use_clean_data and len(stats['normal_values']) > 0:
                self.boundaries['setpoint'] = {'low': stats['normal_min'], 'high': stats['normal_max']}
            else:
                self.boundaries['setpoint'] = {'low': stats['min'], 'high': stats['max']}
            print(f"Setpoint boundaries: {self.boundaries['setpoint']}")
        
        # Time intervals - K-means 2 clusters + outlier (Paper Table III)
        if 'time_intervals' in self.feature_statistics and len(self.feature_statistics['time_intervals']['all_values']) > 0:
            stats = self.feature_statistics['time_intervals']
            time_data = stats['normal_values'].reshape(-1, 1) if use_clean_data and len(stats['normal_values']) > 0 else stats['all_values'].reshape(-1, 1)
            
            if len(time_data) >= 2:
                kmeans_time = KMeans(n_clusters=2, random_state=42)
                kmeans_time.fit(time_data)
                centers = sorted(kmeans_time.cluster_centers_.flatten())
                self.boundaries['time_interval'] = {'low': centers[0], 'high': centers[1]}
                print(f"Time interval clusters: {centers}")
        
        # CRC rate - K-means 2 clusters + outlier (Paper Table III)
        if 'crc_rate' in self.feature_statistics and len(self.feature_statistics['crc_rate']['all_values']) > 0:
            stats = self.feature_statistics['crc_rate']
            crc_data = stats['normal_values'].reshape(-1, 1) if use_clean_data and len(stats['normal_values']) > 0 else stats['all_values'].reshape(-1, 1)
            
            if len(crc_data) >= 2:
                kmeans_crc = KMeans(n_clusters=2, random_state=42)
                kmeans_crc.fit(crc_data)
                centers = sorted(kmeans_crc.cluster_centers_.flatten())
                self.boundaries['crc_rate'] = {'low': centers[0], 'high': centers[1]}
                print(f"CRC rate clusters: {centers}")
        
        # PID parameters - K-means 32 clusters + outlier (Paper Table III)
        pid_params = self._extract_pid_parameters(data)
        if len(pid_params) >= 32:
            self.pid_clusters = KMeans(n_clusters=32, random_state=42)
            self.pid_clusters.fit(pid_params)
            print(f"PID parameters clustered into 32 groups")
        elif len(pid_params) > 0:
            # Use fewer clusters if we don't have enough data
            n_clusters = min(32, len(pid_params))
            self.pid_clusters = KMeans(n_clusters=n_clusters, random_state=42)
            self.pid_clusters.fit(pid_params)
            print(f"PID parameters clustered into {n_clusters} groups")
        
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
        """Create numerical feature vector for LSTM"""
        feature_vector = []
        
        # Add raw numerical features
        raw_features = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10]  # indices of raw features
        for idx in raw_features:
            if len(row) > idx and row[idx] != '?':
                try:
                    feature_vector.append(float(row[idx]))
                except (ValueError, TypeError):
                    feature_vector.append(0.0)
            else:
                feature_vector.append(0.0)
        
        # Add binary features
        if len(row) > 11 and row[11] != '?':
            feature_vector.append(1.0 if row[11] == '1' else 0.0)
        else:
            feature_vector.append(0.0)
            
        if len(row) > 12 and row[12] != '?':
            feature_vector.append(1.0 if row[12] == '1' else 0.0)
        else:
            feature_vector.append(0.0)
        
        # Add discretized numerical features
        if len(row) > 3 and row[3] != '?':  # setpoint
            try:
                setpoint_cat = self.discretize_setpoint(float(row[3]))
                feature_vector.append(self._safe_encode(setpoint_cat))
            except (ValueError, TypeError):
                feature_vector.append(0.0)
        else:
            feature_vector.append(0.0)
            
        if len(row) > 13 and row[13] != '?':  # pressure
            try:
                pressure_cat = self.discretize_pressure(float(row[13]))
                feature_vector.append(self._safe_encode(pressure_cat))
            except (ValueError, TypeError):
                feature_vector.append(0.0)
        else:
            feature_vector.append(0.0)
            
        if len(row) > 14 and row[14] != '?':  # crc_rate
            try:
                crc_cat = self.discretize_crc_rate(float(row[14]))
                feature_vector.append(self._safe_encode(crc_cat))
            except (ValueError, TypeError):
                feature_vector.append(0.0)
        else:
            feature_vector.append(0.0)
        
        # Add command response
        if len(row) > 15 and row[15] != '?':
            feature_vector.append(1.0 if row[15] == '1' else 0.0)
        else:
            feature_vector.append(0.0)
        
        return np.array(feature_vector)
    
    def _safe_encode(self, category_string):
        """Safely convert category to numerical value"""
        try:
            if 'outlier' in category_string:
                return 0.0 if 'low' in category_string else 999.0
            elif 'unknown' in category_string:
                return -1.0
            else:
                parts = category_string.split('_')
                for part in parts:
                    if part.isdigit():
                        return float(part)
                return float(hash(category_string) % 1000)
        except:
            return -1.0
    
    def discretize_dataset(self, data):
        """Discretize entire dataset"""
        print("Generating signatures and feature vectors...")
        
        signatures = []
        feature_vectors = []
        prev_timestamp = None
        
        for i, row in enumerate(data):
            if i % 10000 == 0:
                print(f"  Processed {i}/{len(data)} packets...")
            
            signature = self.generate_signature(row, prev_timestamp)
            signatures.append(signature)
            
            feature_vector = self.create_feature_vector(row)
            feature_vectors.append(feature_vector)
            
            if len(row) > 16 and row[16] != '?':
                try:
                    prev_timestamp = float(row[16])
                except (ValueError, TypeError):
                    pass
        
        unique_signatures = set(signatures)
        print(f"Generated {len(signatures)} signatures")
        print(f"Unique signatures: {len(unique_signatures)}")
        print(f"Feature vector dimension: {len(feature_vectors[0])}")
        
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


class PaperLSTM(nn.Module):
    """Paper-compliant LSTM model: 2 layers with 256 units each + Softmax"""
    
    def __init__(self, input_size, hidden_size, num_classes, num_layers=2):
        super(PaperLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Paper architecture: 2 LSTM layers with 256 units each
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # Paper output: Softmax classifier with num_classes = unique signatures
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # LSTM forward pass
        out, _ = self.lstm(x)
        
        # Take the last time step output
        out = out[:, -1, :]
        
        # Softmax classification
        out = self.fc(out)
        return out


class PaperLSTMDetector:
    """Paper-compliant LSTM implementation using PyTorch"""
    
    def __init__(self, sequence_length=10, input_features=15, lstm_units=256, num_signatures=613):
        self.sequence_length = sequence_length
        self.input_features = input_features
        self.lstm_units = lstm_units
        self.num_signatures = num_signatures
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.signature_to_index = {}
        self.index_to_signature = {}
        
        print(f"Using device: {self.device}")
    
    def build_model(self):
        """Build exact paper architecture: 2 LSTM layers (256 units each) + Softmax"""
        self.model = PaperLSTM(
            input_size=self.input_features,
            hidden_size=self.lstm_units,
            num_classes=self.num_signatures,
            num_layers=2
        ).to(self.device)
        
        print("LSTM Model Architecture:")
        print(f"  Input: ({self.sequence_length}, {self.input_features})")
        print(f"  LSTM Layers: 2")
        print(f"  LSTM Units per layer: {self.lstm_units}")
        print(f"  Output: {self.num_signatures} classes (unique signatures)")
        
        return self.model
    
    def create_signature_mapping(self, unique_signatures):
        self.signature_to_index = {sig: idx for idx, sig in enumerate(unique_signatures)}
        self.index_to_signature = {idx: sig for sig, idx in self.signature_to_index.items()}
        return self.signature_to_index
    
    def create_sequences(self, feature_vectors, signatures, signature_mapping):
        X_sequences = []
        y_sequences = []
        
        for i in range(len(feature_vectors) - self.sequence_length):
            sequence = feature_vectors[i:i + self.sequence_length]
            target_signature = signatures[i + self.sequence_length]
            target_index = signature_mapping.get(target_signature, -1)
            
            if target_index != -1:
                X_sequences.append(sequence)
                y_sequences.append(target_index)
        
        if len(X_sequences) == 0:
            print("Warning: No sequences created. Check if signatures are properly mapped.")
            return np.array([]), np.array([])
            
        X_sequences = np.array(X_sequences)
        y_sequences = np.array(y_sequences)
        
        print(f"Created {len(X_sequences)} training sequences")
        return X_sequences, y_sequences
    
    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        """Train with paper parameters"""
        if len(X_train) == 0:
            print("Error: No training data available.")
            return {'train_loss': [], 'val_accuracy': []}
            
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.LongTensor(y_train).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        y_val_tensor = torch.LongTensor(y_val).to(self.device)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        
        # Paper optimizer: Adam with learning rate 0.001
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        history = {'train_loss': [], 'val_accuracy': []}
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            total_loss = 0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_tensor)
                _, predicted = torch.max(val_outputs, 1)
                val_accuracy = (predicted == y_val_tensor).float().mean().item()
            
            history['train_loss'].append(total_loss / len(train_loader))
            history['val_accuracy'].append(val_accuracy)
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}, Val Acc: {val_accuracy:.4f}')
        
        return history
    
    def evaluate(self, X_test, y_test):
        """Evaluate model on test set"""
        if len(X_test) == 0:
            return 0.0
            
        X_test_tensor = torch.FloatTensor(X_test).to(self.device)
        y_test_tensor = torch.LongTensor(y_test).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_test_tensor)
            _, predicted = torch.max(outputs, 1)
            accuracy = (predicted == y_test_tensor).float().mean().item()
        
        return accuracy
    
    def time_series_anomaly_detection(self, input_sequence, actual_signature, top_k=4):
        """Time-series anomaly detection as per paper"""
        if len(input_sequence) == 0:
            return True, 0.0, []
            
        # Convert to tensor
        if len(input_sequence.shape) == 2:
            input_sequence = np.expand_dims(input_sequence, axis=0)
        
        input_tensor = torch.FloatTensor(input_sequence).to(self.device)
        
        # Get predictions
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        
        # Get top-k predictions
        top_indices = np.argsort(probabilities)[-top_k:][::-1]
        top_probabilities = probabilities[top_indices]
        top_signatures = [self.index_to_signature.get(idx, "unknown") for idx in top_indices]
        
        # Check for anomaly
        is_anomaly = actual_signature not in top_signatures
        
        # Get confidence for actual signature
        actual_index = self.signature_to_index.get(actual_signature, -1)
        if actual_index != -1:
            actual_probability = probabilities[actual_index]
        else:
            actual_probability = 0.0
        
        return is_anomaly, actual_probability, list(zip(top_signatures, top_probabilities))


# Main execution
if __name__ == "__main__":
    # Check if PyTorch is available
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    
    # Initialize the complete system
    ics_analyzer = PaperCompliantICSAnalyzer()
    
    # Run complete pipeline
    arff_file = "Dataset.arff"  # Replace with your dataset path
    try:
        bloom_filter, lstm_detector = ics_analyzer.run_complete_pipeline(arff_file)
        
        print("\nSystem ready for anomaly detection!")
        print("Use ics_analyzer.detect_anomaly(packet_data, previous_packets) for real-time detection")
        
    except FileNotFoundError:
        print(f"Error: Dataset file '{arff_file}' not found.")
        print("Please ensure the dataset file exists in the current directory.")
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()
        print("Please check your dataset format and try again.")