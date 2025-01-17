import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from model import CSILocalizationNet
from dataloader import create_dataloaders
from prepare_dataset import HALOCDataset

def load_baseline_results():
    """Load baseline results from results_summary.md."""
    with open('results_summary.md', 'r') as f:
        content = f.read()
    
    # Extract baseline metrics
    baseline = {
        'x_error': 2.9833,  # meters
        'y_error': 0.3173,  # meters
        'z_error': 0.0119,  # meters
        'x_variance': 3.941,  # from error distribution std
    }
    return baseline

def evaluate_model():
    """Evaluate current model performance."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CSILocalizationNet().to(device)
    
    # Load latest checkpoint
    checkpoint_dirs = sorted(Path('checkpoints').glob('*'))
    if not checkpoint_dirs:
        raise ValueError("No checkpoint directories found")
    latest_dir = checkpoint_dirs[-1]
    
    checkpoint = torch.load(latest_dir / 'best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create test dataloader
    dataloaders = create_dataloaders('HALOC', batch_size=32)
    test_loader = dataloaders['test']
    
    # Collect predictions and ground truth
    all_preds = []
    all_true = []
    
    with torch.no_grad():
        for csi, positions in test_loader:
            csi = csi.to(device)
            outputs = model(csi)
            all_preds.extend(outputs.cpu().numpy())
            all_true.extend(positions.cpu().numpy())
    
    predictions = np.array(all_preds)
    ground_truth = np.array(all_true)
    
    # Calculate errors
    errors = predictions - ground_truth
    mae = np.mean(np.abs(errors), axis=0)
    variance = np.var(errors, axis=0)
    
    return {
        'x_error': mae[0],
        'y_error': mae[1],
        'z_error': mae[2],
        'x_variance': np.sqrt(variance[0])
    }

def plot_improvement_comparison(baseline, current):
    """Plot improvement comparisons."""
    metrics = ['x_error', 'y_error', 'z_error', 'x_variance']
    improvements = {
        metric: ((baseline[metric] - current[metric]) / baseline[metric] * 100)
        for metric in metrics
    }
    
    # Create bar plot
    plt.figure(figsize=(10, 6))
    bars = plt.bar(metrics, [improvements[m] for m in metrics])
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom')
    
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.2)
    plt.title('Improvement Percentages')
    plt.ylabel('Improvement (%)')
    plt.grid(True, alpha=0.3)
    
    # Add target lines
    targets = {
        'x_error': 15,  # Target: 15-20% reduction
        'x_variance': 10  # Target: 10-15% reduction
    }
    for metric, target in targets.items():
        idx = metrics.index(metric)
        plt.axhline(y=target, color='r', linestyle='--', alpha=0.5,
                   xmin=idx/len(metrics), xmax=(idx+1)/len(metrics))
    
    plt.tight_layout()
    plt.savefig('improvement_comparison.png')
    plt.close()
    
    return improvements

def main():
    print("Loading baseline results...")
    baseline = load_baseline_results()
    
    print("\nEvaluating current model...")
    current = evaluate_model()
    
    print("\nComparing Results:")
    print("\nBaseline Metrics:")
    for metric, value in baseline.items():
        print(f"{metric}: {value:.4f}")
    
    print("\nCurrent Metrics:")
    for metric, value in current.items():
        print(f"{metric}: {value:.4f}")
    
    print("\nCalculating improvements...")
    improvements = plot_improvement_comparison(baseline, current)
    
    print("\nImprovement Percentages:")
    targets_met = True
    for metric, improvement in improvements.items():
        print(f"{metric}: {improvement:.1f}%")
        if metric == 'x_error' and improvement < 15:
            targets_met = False
            print("! Below target: Expected 15-20% reduction in X-axis error")
        elif metric == 'x_variance' and improvement < 10:
            targets_met = False
            print("! Below target: Expected 10-15% reduction in variance")
    
    print(f"\nAll targets met: {targets_met}")
    print("Results visualization saved as 'improvement_comparison.png'")

if __name__ == '__main__':
    main()
