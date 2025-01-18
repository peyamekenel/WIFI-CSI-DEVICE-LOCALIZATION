import torch
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # This import registers the 3D projection
import sys
import os

# Add HALOC repo to path for using their dataset class
sys.path.append('/home/ubuntu/HALOC')
from datasets import HALOC
from model import HALOCNet

def load_best_model(model_path='best_model.pth', device='cpu'):
    """Load the best model checkpoint."""
    model = HALOCNet().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def visualize_predictions(true_positions, predicted_positions, save_path='prediction_visualization.png'):
    """Create 3D scatter plot of true vs predicted positions."""
    try:
        # Create figure with larger size and higher DPI for better quality
        fig = plt.figure(figsize=(15, 10), dpi=150)
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot a subset of points for better visibility (every 10th point)
        stride = 10
        true_subset = true_positions[::stride]
        pred_subset = predicted_positions[::stride]
        
        # Plot true positions
        ax.scatter(true_subset[:, 0], 
                  true_subset[:, 1], 
                  true_subset[:, 2], 
                  c='blue', 
                  marker='o', 
                  label='True Positions')
        
        # Plot predicted positions
        ax.scatter(pred_subset[:, 0], 
                  pred_subset[:, 1], 
                  pred_subset[:, 2], 
                  c='red', 
                  marker='^', 
                  label='Predicted Positions')
        
        # Add labels and title
        ax.set(xlabel='X Position (m)',
               ylabel='Y Position (m)',
               zlabel='Z Position (m)',
               title='True vs Predicted 3D Positions')
        
        # Add legend and grid
        ax.legend()
        ax.grid(True)
        
        # Save the plot with high quality
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"Error creating 3D scatter plot: {str(e)}")
        raise

def calculate_metrics(true_positions, predicted_positions):
    """Calculate error metrics for the predictions."""
    # Calculate MSE per dimension
    mse_per_dim = np.mean((true_positions - predicted_positions) ** 2, axis=0)
    
    # Calculate RMSE per dimension
    rmse_per_dim = np.sqrt(mse_per_dim)
    
    # Calculate Euclidean distance error
    euclidean_errors = np.sqrt(np.sum((true_positions - predicted_positions) ** 2, axis=1))
    mean_euclidean_error = np.mean(euclidean_errors)
    median_euclidean_error = np.median(euclidean_errors)
    
    return {
        'mse_per_dimension': mse_per_dim,
        'rmse_per_dimension': rmse_per_dim,
        'mean_euclidean_error': mean_euclidean_error,
        'median_euclidean_error': median_euclidean_error,
        'euclidean_errors': euclidean_errors
    }

def plot_error_distribution(errors, save_path='error_distribution.png'):
    """Plot histogram of prediction errors."""
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=50, edgecolor='black')
    plt.xlabel('Euclidean Error (m)')
    plt.ylabel('Frequency')
    plt.title('Distribution of 3D Position Prediction Errors')
    plt.savefig(save_path)
    plt.close()

def evaluate_model(model_path='best_model.pth', test_data_path='/home/ubuntu/datasets/haloc/HALOC/5.csv', max_samples=1000):
    """Evaluate the model on test data and generate visualizations."""
    print("\nStarting model evaluation...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load model
    print("Loading model...")
    model = load_best_model(model_path, device)
    
    # Load test dataset
    print("Loading test dataset...")
    test_dataset = HALOC(test_data_path, windowSize=351)
    total_samples = len(test_dataset)
    eval_samples = min(max_samples, total_samples)
    print(f"Test dataset size: {total_samples} samples")
    print(f"Using first {eval_samples} samples for evaluation")
    
    # Collect predictions
    print("\nGenerating predictions...")
    true_positions = []
    predicted_positions = []
    
    try:
        with torch.no_grad():
            for i in tqdm(range(eval_samples), desc="Processing test samples"):
                features, labels = test_dataset[i]
                features = features.unsqueeze(0).float().to(device)
                prediction = model(features)
                
                true_positions.append(labels.numpy())
                predicted_positions.append(prediction.cpu().numpy()[0])
        
        true_positions = np.array(true_positions)
        predicted_positions = np.array(predicted_positions)
        
        # Calculate metrics
        print("\nCalculating metrics...")
        metrics = calculate_metrics(true_positions, predicted_positions)
        
        # Generate visualizations
        print("\nGenerating visualizations...")
        print("Creating 3D scatter plot...")
        visualize_predictions(true_positions, predicted_positions)
        print("Creating error distribution plot...")
        plot_error_distribution(metrics['euclidean_errors'])
        
        # Print metrics
        print("\nModel Evaluation Metrics:")
        print("-" * 30)
        print(f"MSE per dimension (x,y,z):")
        for i, dim in enumerate(['x', 'y', 'z']):
            print(f"  {dim}: {metrics['mse_per_dimension'][i]:.6f}")
        print(f"\nRMSE per dimension (x,y,z):")
        for i, dim in enumerate(['x', 'y', 'z']):
            print(f"  {dim}: {metrics['rmse_per_dimension'][i]:.6f}")
        print(f"\nMean Euclidean Error: {metrics['mean_euclidean_error']:.4f} m")
        print(f"Median Euclidean Error: {metrics['median_euclidean_error']:.4f} m")
        
        print("\nEvaluation completed successfully!")
        print(f"Plots saved as 'prediction_visualization.png' and 'error_distribution.png'")
        
        return metrics
        
    except Exception as e:
        print(f"\nError during evaluation: {str(e)}")
        raise

if __name__ == '__main__':
    evaluate_model()
