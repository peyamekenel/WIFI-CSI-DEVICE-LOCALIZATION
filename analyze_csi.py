import numpy as np
import matplotlib.pyplot as plt
from prepare_dataset import HALOCDataset
import pandas as pd

def plot_csi_characteristics(raw_csi, processed_csi, sample_idx=0):
    """Plot characteristics of raw vs processed CSI data."""
    # Create subplots
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle(f'CSI Characteristics Analysis - Sample {sample_idx}')
    
    # Raw CSI analysis
    raw_mag = np.abs(raw_csi)
    raw_phase = np.angle(raw_csi)
    raw_real = raw_csi.real
    
    # Processed CSI analysis
    proc_mag = np.abs(processed_csi)
    proc_phase = np.angle(processed_csi)
    proc_real = processed_csi.real
    
    # Plot magnitudes
    axes[0,0].plot(raw_mag, label='Raw')
    axes[0,0].set_title('Magnitude (Raw)')
    axes[0,0].set_ylabel('Magnitude')
    axes[0,0].grid(True)
    
    axes[0,1].plot(proc_mag, label='Processed')
    axes[0,1].set_title('Magnitude (Processed)')
    axes[0,1].grid(True)
    
    # Plot phases
    axes[1,0].plot(raw_phase, label='Raw')
    axes[1,0].set_title('Phase (Raw)')
    axes[1,0].set_ylabel('Phase (radians)')
    axes[1,0].grid(True)
    
    axes[1,1].plot(proc_phase, label='Processed')
    axes[1,1].set_title('Phase (Processed)')
    axes[1,1].grid(True)
    
    # Plot real parts
    axes[2,0].plot(raw_real, label='Raw')
    axes[2,0].set_title('Real Component (Raw)')
    axes[2,0].set_ylabel('Real Value')
    axes[2,0].grid(True)
    
    axes[2,1].plot(proc_real, label='Processed')
    axes[2,1].set_title('Real Component (Processed)')
    axes[2,1].grid(True)
    
    # Add statistics
    stats_raw = f'Raw Stats:\nMag Range: [{raw_mag.min():.2f}, {raw_mag.max():.2f}]\n'
    stats_raw += f'Phase Range: [{raw_phase.min():.2f}, {raw_phase.max():.2f}]\n'
    stats_raw += f'Real Range: [{raw_real.min():.2f}, {raw_real.max():.2f}]'
    
    stats_proc = f'Processed Stats:\nMag Range: [{proc_mag.min():.2f}, {proc_mag.max():.2f}]\n'
    stats_proc += f'Phase Range: [{proc_phase.min():.2f}, {proc_phase.max():.2f}]\n'
    stats_proc += f'Real Range: [{proc_real.min():.2f}, {proc_real.max():.2f}]'
    
    fig.text(0.15, 0.02, stats_raw, fontsize=10)
    fig.text(0.65, 0.02, stats_proc, fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f'csi_analysis_{sample_idx}.png')
    plt.close()
    
    return {
        'raw': {
            'mag_range': (raw_mag.min(), raw_mag.max()),
            'phase_range': (raw_phase.min(), raw_phase.max()),
            'real_range': (raw_real.min(), raw_real.max())
        },
        'processed': {
            'mag_range': (proc_mag.min(), proc_mag.max()),
            'phase_range': (proc_phase.min(), proc_phase.max()),
            'real_range': (proc_real.min(), proc_real.max())
        }
    }

def analyze_csi_data():
    """Analyze CSI data characteristics before and after processing."""
    dataset = HALOCDataset('HALOC')
    
    # Load test samples
    test_data = dataset.load_split('test')
    
    # Analyze multiple samples
    num_samples = 5
    all_stats = []
    
    print("\nAnalyzing CSI characteristics...")
    for i in range(num_samples):
        sample = test_data.iloc[i]
        
        # Get raw CSI data
        raw_csi = np.array([complex(x) for x in sample['data'].strip('[]').split(',')])
        
        # Get processed CSI data
        processed_csi = dataset.process_csi_data(sample['data'])
        
        # Plot and get statistics
        stats = plot_csi_characteristics(raw_csi, processed_csi, i)
        all_stats.append(stats)
        
        print(f"\nSample {i} Analysis:")
        print("Raw CSI - Magnitude range:", stats['raw']['mag_range'])
        print("Processed CSI - Magnitude range:", stats['processed']['mag_range'])
        print("Phase range change:", 
              f"{stats['raw']['phase_range']} -> {stats['processed']['phase_range']}")
    
    print("\nVisualization files created:")
    for i in range(num_samples):
        print(f"- csi_analysis_{i}.png")

if __name__ == '__main__':
    analyze_csi_data()
