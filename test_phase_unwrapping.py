import numpy as np
import matplotlib.pyplot as plt
from prepare_dataset import HALOCDataset
import pandas as pd

import time

def test_phase_unwrapping():
    """Test the phase unwrapping implementation."""
    dataset = HALOCDataset('HALOC')
    
    # Load multiple samples for better testing
    df = pd.read_csv('HALOC/5.csv', nrows=1000)
    
    # Measure processing time
    start_time = time.time()
    
    # Process multiple samples to get better statistics
    original_phases = []
    unwrapped_phases = []
    processing_times = []
    
    for idx in range(min(100, len(df))):
        sample = df.iloc[idx]
        
        # Time the processing
        sample_start = time.time()
        csi_array = dataset.process_csi_data(sample['data'])
        processing_times.append(time.time() - sample_start)
        
        # Get original and unwrapped phases
        original_phase = np.angle(np.array([complex(x) for x in sample['data'].strip('[]').split(',')]))
        unwrapped_phase = np.angle(csi_array)
        
        original_phases.append(original_phase)
        unwrapped_phases.append(unwrapped_phase)
    
    total_time = time.time() - start_time
    
    # Convert to numpy arrays for analysis
    original_phases = np.array(original_phases)
    unwrapped_phases = np.array(unwrapped_phases)
    
    # Plot comparison for a representative sample
    plt.figure(figsize=(12, 6))
    plt.plot(original_phases[0], label='Original Phase', alpha=0.7)
    plt.plot(unwrapped_phases[0], label='Unwrapped Phase', alpha=0.7)
    plt.xlabel('Subcarrier Index')
    plt.ylabel('Phase (radians)')
    plt.title('Phase Unwrapping Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig('phase_unwrapping_test.png')
    plt.close()
    
    # Analyze phase characteristics
    phase_diffs_orig = np.diff(original_phases, axis=1)
    phase_diffs_unwrap = np.diff(unwrapped_phases, axis=1)
    
    # Detect sharp transitions (potential discontinuities)
    threshold = np.pi/2
    orig_jumps = np.abs(phase_diffs_orig) > threshold
    unwrap_jumps = np.abs(phase_diffs_unwrap) > threshold
    
    # Calculate statistics
    orig_discontinuities = np.sum(orig_jumps, axis=1)
    unwrap_discontinuities = np.sum(unwrap_jumps, axis=1)
    
    print("\nPhase Unwrapping Analysis:")
    print(f"Original Phase - Range: [{np.min(original_phases):.2f}, {np.max(original_phases):.2f}]")
    print(f"Unwrapped Phase - Range: [{np.min(unwrapped_phases):.2f}, {np.max(unwrapped_phases):.2f}]")
    
    print(f"\nPhase Transition Analysis:")
    print(f"Mean phase difference (original): {np.mean(np.abs(phase_diffs_orig)):.3f} rad")
    print(f"Mean phase difference (unwrapped): {np.mean(np.abs(phase_diffs_unwrap)):.3f} rad")
    
    print(f"\nDiscontinuity Statistics (threshold = {threshold:.2f} rad):")
    print(f"Average discontinuities per sample (original): {np.mean(orig_discontinuities):.1f}")
    print(f"Average discontinuities per sample (unwrapped): {np.mean(unwrap_discontinuities):.1f}")
    
    if np.mean(orig_discontinuities) > 0:
        reduction = 100 * (1 - np.mean(unwrap_discontinuities)/np.mean(orig_discontinuities))
        print(f"Discontinuity reduction: {reduction:.1f}%")
    
    # Plot phase differences
    plt.figure(figsize=(12, 8))
    plt.subplot(211)
    plt.plot(phase_diffs_orig[0], label='Original', alpha=0.7)
    plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
    plt.axhline(y=-threshold, color='r', linestyle='--')
    plt.title('Phase Differences (Original)')
    plt.ylabel('Phase Difference (rad)')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(212)
    plt.plot(phase_diffs_unwrap[0], label='Unwrapped', alpha=0.7)
    plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
    plt.axhline(y=-threshold, color='r', linestyle='--')
    plt.title('Phase Differences (Unwrapped)')
    plt.xlabel('Subcarrier Index')
    plt.ylabel('Phase Difference (rad)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('phase_differences.png')
    plt.close()
    
    print(f"\nPerformance Analysis:")
    print(f"Average processing time per sample: {np.mean(processing_times)*1000:.2f} ms")
    print(f"Total processing time for {len(processing_times)} samples: {total_time:.2f} s")

if __name__ == '__main__':
    test_phase_unwrapping()
