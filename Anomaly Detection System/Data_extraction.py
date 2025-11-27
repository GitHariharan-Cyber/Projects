import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
import matplotlib.pyplot as plt
import re

class PaperCompliantDiscretizer:
    def __init__(self):
        self.boundaries = {}
        self.pid_clusters = None
        self.feature_statistics = {}
        
    def load_arff_data(self, file_path):
        """Load ARFF format dataset"""
        data = []
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # Skip header and find data section
        in_data_section = False
        for line in lines:
            line = line.strip()
            if line.lower() == '@data':
                in_data_section = True
                continue
            if in_data_section and line and not line.startswith('@'):
                # Split by comma and remove quotes if any
                row = [x.strip().strip("'") for x in line.split(',')]
                data.append(row)
        
        print(f"Loaded {len(data)} samples from {file_path}")
        return data
    
    def analyze_feature_statistics(self, data):
        """Analyze dataset for paper-specified features only"""
        print("Analyzing dataset statistics for paper-specified features...")
        
        # Extract only paper-specified features (Table III)
        features = {
            'setpoint': [], 'pressure': [], 'crc_rate': [], 'time_intervals': [], 'pid_params': []
        }
        
        # Collect timestamp for time intervals
        timestamps = []
        for row in data:
            if len(row) > 16 and row[16] != '?':
                timestamps.append(float(row[16]))
        
        # Calculate time intervals
        for i in range(1, len(timestamps)):
            features['time_intervals'].append(timestamps[i] - timestamps[i-1])
        
        # Extract paper-specified features
        for row in data:
            # Setpoint (index 3)
            if len(row) > 3 and row[3] != '?':
                features['setpoint'].append(float(row[3]))
            
            # Pressure (index 13)
            if len(row) > 13 and row[13] != '?':
                features['pressure'].append(float(row[13]))
            
            # CRC rate (index 14)
            if len(row) > 14 and row[14] != '?':
                features['crc_rate'].append(float(row[14]))
            
            # PID parameters (indices 4,5,6,7,8)
            pid_params = []
            valid_pid = True
            for idx in [4, 5, 6, 7, 8]:
                if len(row) <= idx or row[idx] == '?':
                    valid_pid = False
                    break
                pid_params.append(float(row[idx]))
            
            if valid_pid and len(pid_params) == 5:
                features['pid_params'].append(pid_params)
        
        # Analyze each paper-specified feature
        self.feature_statistics = {}
        for feature_name, values in features.items():
            if values and feature_name != 'pid_params' and all(isinstance(x, (int, float)) for x in values):
                values = np.array(values)
                stats_dict = self._calculate_robust_statistics(values, feature_name)
                self.feature_statistics[feature_name] = stats_dict
        
        return self.feature_statistics
    
    def _calculate_robust_statistics(self, values, feature_name):
        """Calculate statistics with outlier detection for paper-specified features"""
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
        
        # Calculate IQR and outlier bounds (Tukey's fences)
        iqr = stats_dict['q3'] - stats_dict['q1']
        lower_bound = stats_dict['q1'] - 1.5 * iqr
        upper_bound = stats_dict['q3'] + 1.5 * iqr
        
        # Identify outliers
        outliers = values[(values < lower_bound) | (values > upper_bound)]
        normal_values = values[(values >= lower_bound) & (values <= upper_bound)]
        
        stats_dict.update({
            'outliers_count': len(outliers),
            'outliers_percentage': len(outliers) / len(values) * 100,
            'normal_min': np.min(normal_values) if len(normal_values) > 0 else lower_bound,
            'normal_max': np.max(normal_values) if len(normal_values) > 0 else upper_bound,
            'outlier_lower_bound': lower_bound,
            'outlier_upper_bound': upper_bound,
            'normal_values': normal_values
        })
        
        print(f"{feature_name:15} | Normal: {stats_dict['normal_min']:8.3f} - {stats_dict['normal_max']:8.3f} | "
              f"Outliers: {stats_dict['outliers_count']:3d} ({stats_dict['outliers_percentage']:5.1f}%)")
        
        return stats_dict
    
    def fit(self, data, use_clean_data=True):
        """Calculate discretization boundaries for paper-specified features only"""
        print("\n" + "="*60)
        print("CALCULATING DISCRETIZATION BOUNDARIES (PAPER COMPLIANT)")
        print("="*60)
        
        # Analyze only paper-specified features
        self.analyze_feature_statistics(data)
        
        # Pressure - Even intervals (20 categories + outlier) - PAPER SPECIFIED
        if 'pressure' in self.feature_statistics:
            stats = self.feature_statistics['pressure']
            if use_clean_data and stats['outliers_count'] > 0:
                print(f"Pressure: Using CLEAN data range (excluding {stats['outliers_count']} outliers)")
                self.boundaries['pressure'] = {
                    'low': stats['normal_min'],
                    'high': stats['normal_max']
                }
            else:
                print("Pressure: Using FULL data range (including outliers)")
                self.boundaries['pressure'] = {
                    'low': stats['min'],
                    'high': stats['max']
                }
        
        # Setpoint - Even intervals (10 categories + outlier) - PAPER SPECIFIED
        if 'setpoint' in self.feature_statistics:
            stats = self.feature_statistics['setpoint']
            if use_clean_data and stats['outliers_count'] > 0:
                self.boundaries['setpoint'] = {
                    'low': stats['normal_min'],
                    'high': stats['normal_max']
                }
            else:
                self.boundaries['setpoint'] = {
                    'low': stats['min'],
                    'high': stats['max']
                }
        
        # Time intervals - K-means (2 clusters + outlier) - PAPER SPECIFIED
        if 'time_intervals' in self.feature_statistics:
            stats = self.feature_statistics['time_intervals']
            time_data = stats['normal_values'].reshape(-1, 1) if use_clean_data and len(stats['normal_values']) > 0 else stats['all_values'].reshape(-1, 1)
            
            if len(time_data) >= 2:
                kmeans_time = KMeans(n_clusters=2, random_state=42)
                kmeans_time.fit(time_data)
                centers = sorted(kmeans_time.cluster_centers_.flatten())
                self.boundaries['time_interval'] = {
                    'low': centers[0],
                    'high': centers[1]
                }
                print(f"Time intervals - Cluster centers: {centers}")
        
        # CRC rate - K-means (2 clusters + outlier) - PAPER SPECIFIED
        if 'crc_rate' in self.feature_statistics:
            stats = self.feature_statistics['crc_rate']
            crc_data = stats['normal_values'].reshape(-1, 1) if use_clean_data and len(stats['normal_values']) > 0 else stats['all_values'].reshape(-1, 1)
            
            if len(crc_data) >= 2:
                kmeans_crc = KMeans(n_clusters=2, random_state=42)
                kmeans_crc.fit(crc_data)
                centers = sorted(kmeans_crc.cluster_centers_.flatten())
                self.boundaries['crc_rate'] = {
                    'low': centers[0],
                    'high': centers[1]
                }
                print(f"CRC rates - Cluster centers: {centers}")
        
        # PID parameters - K-means (32 clusters + outlier) - PAPER SPECIFIED
        pid_params = self._extract_pid_parameters(data)
        if len(pid_params) >= 32:
            self.pid_clusters = KMeans(n_clusters=32, random_state=42)
            self.pid_clusters.fit(pid_params)
            print(f"PID parameters - Clustered {len(pid_params)} samples into 32 groups")
        
        print("\nPaper-compliant discretization boundaries calculated successfully!")
        return self
    
    def _extract_pid_parameters(self, data):
        """Extract PID parameters for clustering"""
        pid_params = []
        for row in data:
            pid_array = []
            valid_pid = True
            for idx in [4, 5, 6, 7, 8]:  # gain, reset_rate, deadband, cycle_time, rate
                if len(row) <= idx or row[idx] == '?':
                    valid_pid = False
                    break
                pid_array.append(float(row[idx]))
            
            if valid_pid and len(pid_array) == 5:
                pid_params.append(pid_array)
        
        return np.array(pid_params)
    
    def discretize_pressure(self, value):
        """Discretize pressure into 20 categories + outliers - PAPER SPECIFIED"""
        if 'pressure' not in self.boundaries:
            return 'pressure_unknown'
        
        bounds = self.boundaries['pressure']
        
        # Check for outliers
        if value < bounds['low']:
            return 'pressure_low_outlier'
        elif value > bounds['high']:
            return 'pressure_high_outlier'
        
        # Divide normal range into 20 equal intervals
        interval_width = (bounds['high'] - bounds['low']) / 20
        for i in range(20):
            lower_bound = bounds['low'] + i * interval_width
            upper_bound = bounds['low'] + (i + 1) * interval_width
            
            if lower_bound <= value < upper_bound:
                return f'pressure_{i+1:02d}'
        
        return 'pressure_20'
    
    def discretize_setpoint(self, value):
        """Discretize setpoint into 10 categories + outliers - PAPER SPECIFIED"""
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
        """Discretize time interval using K-means clusters - PAPER SPECIFIED"""
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
        """Discretize CRC rate using K-means clusters - PAPER SPECIFIED"""
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
        """Discretize PID parameters using K-means clusters - PAPER SPECIFIED"""
        if self.pid_clusters is None:
            return 'pid_unknown'
        
        pid_array = np.array(pid_array).reshape(1, -1)
        cluster_idx = self.pid_clusters.predict(pid_array)[0]
        return f'pid_cluster_{cluster_idx+1:02d}'
    
    def _safe_numerical_encoding(self, category_string, prefix):
        """Safely convert category string to numerical value"""
        try:
            if 'outlier' in category_string:
                # Outliers get special encoding
                if 'low' in category_string:
                    return 0  # Low outlier
                elif 'high' in category_string:
                    return 999  # High outlier
                else:
                    return 998  # Generic outlier
            elif 'unknown' in category_string:
                return -1  # Unknown value
            else:
                # Extract number from string like 'pressure_05' or 'setpoint_03'
                parts = category_string.split('_')
                for part in parts:
                    if part.isdigit():
                        return int(part)
                # If no number found, use hash-based encoding
                return hash(category_string) % 1000
        except:
            return -1  # Default for any error
    
    def generate_signature(self, row, prev_timestamp=None):
        """Generate signature for Bloom Filter - STRICT PAPER COMPLIANCE"""
        signature_parts = []
        
        # 1. Non-specified features - keep as raw values
        if len(row) > 0 and row[0] != '?':
            signature_parts.append(f"addr_{row[0]}")  # Keep raw
        else:
            signature_parts.append("addr_unknown")
            
        if len(row) > 1 and row[1] != '?':
            signature_parts.append(f"func_{row[1]}")  # Keep raw
        else:
            signature_parts.append("func_unknown")
        
        # 2. Length - keep as raw value (paper doesn't specify discretization)
        if len(row) > 2 and row[2] != '?':
            signature_parts.append(f"len_{row[2]}")  # Raw length value
        else:
            signature_parts.append("len_unknown")
        
        # 3. Paper-specified discretization: Setpoint (Table III)
        if len(row) > 3 and row[3] != '?':
            setpoint_val = float(row[3])
            signature_parts.append(self.discretize_setpoint(setpoint_val))
        else:
            signature_parts.append("setpoint_unknown")
        
        # 4. Paper-specified discretization: PID parameters (Table III)
        pid_params = []
        valid_pid = True
        for idx in [4, 5, 6, 7, 8]:
            if len(row) <= idx or row[idx] == '?':
                valid_pid = False
                break
            pid_params.append(float(row[idx]))
        
        if valid_pid and len(pid_params) == 5:
            signature_parts.append(self.discretize_pid_params(pid_params))
        else:
            signature_parts.append("pid_unknown")
        
        # 5. Non-specified features - keep as raw values
        if len(row) > 9 and row[9] != '?':
            signature_parts.append(f"mode_{row[9]}")  # Keep raw
        else:
            signature_parts.append("mode_unknown")
            
        if len(row) > 10 and row[10] != '?':
            signature_parts.append(f"scheme_{row[10]}")  # Keep raw
        else:
            signature_parts.append("scheme_unknown")
        
        if len(row) > 11 and row[11] != '?':
            signature_parts.append(f"pump_{row[11]}")  # Keep raw
        else:
            signature_parts.append("pump_unknown")
            
        if len(row) > 12 and row[12] != '?':
            signature_parts.append(f"solenoid_{row[12]}")  # Keep raw
        else:
            signature_parts.append("solenoid_unknown")
        
        # 6. Paper-specified discretization: Pressure (Table III)
        if len(row) > 13 and row[13] != '?':
            pressure_val = float(row[13])
            signature_parts.append(self.discretize_pressure(pressure_val))
        else:
            signature_parts.append("pressure_unknown")
        
        # 7. Paper-specified discretization: CRC rate (Table III)
        if len(row) > 14 and row[14] != '?':
            crc_val = float(row[14])
            signature_parts.append(self.discretize_crc_rate(crc_val))
        else:
            signature_parts.append("crc_unknown")
        
        # 8. Non-specified feature - keep as raw value
        if len(row) > 15 and row[15] != '?':
            signature_parts.append(f"cmd_{row[15]}")  # Keep raw
        else:
            signature_parts.append("cmd_unknown")
        
        # 9. Paper-specified discretization: Time interval (Table III)
        if prev_timestamp is not None and len(row) > 16 and row[16] != '?':
            current_time = float(row[16])
            time_interval = current_time - prev_timestamp
            signature_parts.append(self.discretize_time_interval(time_interval))
        else:
            signature_parts.append("time_unknown")
        
        return "|".join(signature_parts)
    
    def create_feature_vector(self, row):
        """Create numerical feature vector for LSTM - STRICT PAPER COMPLIANCE"""
        feature_vector = []
        
        # 1. Non-specified features - keep as raw numerical values
        feature_vector.append(float(row[0]) if len(row) > 0 and row[0] != '?' else 0.0)  # address
        feature_vector.append(float(row[1]) if len(row) > 1 and row[1] != '?' else 0.0)  # function
        feature_vector.append(float(row[2]) if len(row) > 2 and row[2] != '?' else 0.0)  # length (RAW)
        
        # 2. Paper-specified features - use discretized numerical representation
        if len(row) > 3 and row[3] != '?':  # setpoint
            setpoint_cat = self.discretize_setpoint(float(row[3]))
            setpoint_val = self._safe_numerical_encoding(setpoint_cat, 'setpoint')
            feature_vector.append(setpoint_val)
        else:
            feature_vector.append(-1)  # Unknown value
        
        # 3. Non-specified features - keep raw
        feature_vector.append(float(row[4]) if len(row) > 4 and row[4] != '?' else 0.0)  # gain
        feature_vector.append(float(row[5]) if len(row) > 5 and row[5] != '?' else 0.0)  # reset_rate
        feature_vector.append(float(row[6]) if len(row) > 6 and row[6] != '?' else 0.0)  # deadband
        feature_vector.append(float(row[7]) if len(row) > 7 and row[7] != '?' else 0.0)  # cycle_time
        feature_vector.append(float(row[8]) if len(row) > 8 and row[8] != '?' else 0.0)  # rate
        
        # 4. Non-specified features - keep raw
        feature_vector.append(float(row[9]) if len(row) > 9 and row[9] != '?' else 0.0)  # system_mode
        feature_vector.append(float(row[10]) if len(row) > 10 and row[10] != '?' else 0.0) # control_scheme
        feature_vector.append(1.0 if len(row) > 11 and row[11] == '1' else 0.0)  # pump
        feature_vector.append(1.0 if len(row) > 12 and row[12] == '1' else 0.0)  # solenoid
        
        # 5. Paper-specified features - discretized
        if len(row) > 13 and row[13] != '?':  # pressure
            pressure_cat = self.discretize_pressure(float(row[13]))
            pressure_val = self._safe_numerical_encoding(pressure_cat, 'pressure')
            feature_vector.append(pressure_val)
        else:
            feature_vector.append(-1)  # Unknown value
        
        if len(row) > 14 and row[14] != '?':  # crc_rate
            crc_cat = self.discretize_crc_rate(float(row[14]))
            crc_val = self._safe_numerical_encoding(crc_cat, 'crc')
            feature_vector.append(crc_val)
        else:
            feature_vector.append(-1)  # Unknown value
        
        # 6. Non-specified feature - keep raw
        feature_vector.append(1.0 if len(row) > 15 and row[15] == '1' else 0.0)  # command_response
        
        return np.array(feature_vector)
    
    def discretize_dataset(self, data):
        """Discretize entire dataset and generate signatures"""
        print("\nGenerating signatures for entire dataset...")
        
        signatures = []
        feature_vectors = []
        prev_timestamp = None
        
        for i, row in enumerate(data):
            if i % 10000 == 0:
                print(f"Processed {i}/{len(data)} packets...")
            
            # Generate signature for Bloom Filter
            signature = self.generate_signature(row, prev_timestamp)
            signatures.append(signature)
            
            # Generate feature vector for LSTM
            feature_vector = self.create_feature_vector(row)
            feature_vectors.append(feature_vector)
            
            if len(row) > 16 and row[16] != '?':
                prev_timestamp = float(row[16])
        
        # Count unique signatures
        unique_signatures = set(signatures)
        print(f"\nGenerated {len(signatures)} signatures")
        print(f"Unique signatures: {len(unique_signatures)}")
        print(f"Feature vector dimension: {len(feature_vectors[0])}")
        
        return signatures, np.array(feature_vectors), unique_signatures

# Main execution
if __name__ == "__main__":
    # Initialize paper-compliant discretizer
    discretizer = PaperCompliantDiscretizer()
    
    # Load your dataset
    data = discretizer.load_arff_data('Dataset.arff')
    
    # Fit discretization boundaries for paper-specified features only
    discretizer.fit(data, use_clean_data=True)
    
    # Generate signatures and feature vectors
    signatures, feature_vectors, unique_signatures = discretizer.discretize_dataset(data)
    
    # Save results
    with open('paper_signatures.txt', 'w') as f:
        for sig in signatures:
            f.write(sig + '\n')
    
    with open('paper_unique_signatures.txt', 'w') as f:
        for sig in sorted(unique_signatures):
            f.write(sig + '\n')
    
    # Save feature vectors for LSTM
    np.save('paper_feature_vectors.npy', feature_vectors)
    
    # Print summary
    print("\n" + "="*60)
    print("PAPER-COMPLIANT DISCRETIZATION COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"Total signatures: {len(signatures)}")
    print(f"Unique signatures: {len(unique_signatures)}")
    print(f"LSTM output layer size: {len(unique_signatures)} neurons")
    print(f"LSTM input dimension: {feature_vectors.shape[1]} features")
    print("\nSample signatures:")
    for i in range(min(3, len(signatures))):
        print(f"{i+1}. {signatures[i]}")
    
    print("\nFiles saved:")
    print("- paper_signatures.txt: All signatures for Bloom Filter")
    print("- paper_unique_signatures.txt: Unique signatures for LSTM output layer") 
    print("- paper_feature_vectors.npy: Numerical features for LSTM input")