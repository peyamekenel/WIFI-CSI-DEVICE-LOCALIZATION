import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def load_dataset():
    """Load the preprocessed and split dataset."""
    print("Loading dataset...")
    with open('csi_dataset.pkl', 'rb') as f:
        dataset = pickle.load(f)
    return dataset['X_train'], dataset['X_test'], dataset['y_train'], dataset['y_test']

def prepare_data(X_train, X_test, y_train, y_test):
    """Prepare data for training by scaling features and creating classification labels."""
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Prepare classification labels (combine distance and angle into categories)
    y_train_class = np.array([f"{d}m_{a}deg" for d, a in y_train])
    y_test_class = np.array([f"{d}m_{a}deg" for d, a in y_test])
    
    # Prepare regression targets (separate distance and angle)
    y_train_reg = y_train
    y_test_reg = y_test
    
    return (X_train_scaled, X_test_scaled, 
            y_train_class, y_test_class,
            y_train_reg, y_test_reg)

def train_classification_models(X_train, X_test, y_train, y_test):
    """Train and evaluate classification models."""
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(kernel='rbf', random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5)
    }
    
    results = {}
    print("\nTraining Classification Models:")
    print("-" * 50)
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = {
            'accuracy': accuracy,
            'predictions': y_pred
        }
        print(f"{name} Accuracy: {accuracy:.4f}")
    
    return results

def train_regression_model(X_train, X_test, y_train, y_test):
    """Train and evaluate regression model for distance and angle prediction."""
    print("\nTraining Regression Model:")
    print("-" * 50)
    
    # Train separate models for distance and angle
    distance_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    angle_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    
    # Train distance model
    print("Training distance predictor...")
    distance_model.fit(X_train, y_train[:, 0])  # First column is distance
    distance_pred = distance_model.predict(X_test)
    distance_mse = mean_squared_error(y_test[:, 0], distance_pred)
    distance_mae = mean_absolute_error(y_test[:, 0], distance_pred)
    
    # Train angle model
    print("Training angle predictor...")
    angle_model.fit(X_train, y_train[:, 1])  # Second column is angle
    angle_pred = angle_model.predict(X_test)
    angle_mse = mean_squared_error(y_test[:, 1], angle_pred)
    angle_mae = mean_absolute_error(y_test[:, 1], angle_pred)
    
    results = {
        'distance': {
            'mse': distance_mse,
            'mae': distance_mae,
            'predictions': distance_pred
        },
        'angle': {
            'mse': angle_mse,
            'mae': angle_mae,
            'predictions': angle_pred
        }
    }
    
    print(f"\nDistance Prediction:")
    print(f"MSE: {distance_mse:.4f}")
    print(f"MAE: {distance_mae:.4f}")
    print(f"\nAngle Prediction:")
    print(f"MSE: {angle_mse:.4f}")
    print(f"MAE: {angle_mae:.4f}")
    
    return results


def plot_results(class_results, reg_results, y_test_class, y_test_reg):
    """Generate visualizations of model performance."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Plot classification results
    plt.figure(figsize=(10, 6))
    accuracies = [results['accuracy'] for results in class_results.values()]
    plt.bar(class_results.keys(), accuracies)
    plt.title('Classification Model Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'model_comparison_{timestamp}.png')
    plt.close()
    
    # Plot regression results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Distance predictions
    ax1.scatter(y_test_reg[:, 0], reg_results['distance']['predictions'], alpha=0.5)
    ax1.plot([1, 5], [1, 5], 'r--')  # Perfect prediction line
    ax1.set_xlabel('True Distance (m)')
    ax1.set_ylabel('Predicted Distance (m)')
    ax1.set_title('Distance Prediction Performance')
    
    # Angle predictions
    ax2.scatter(y_test_reg[:, 1], reg_results['angle']['predictions'], alpha=0.5)
    ax2.plot([-60, 30], [-60, 30], 'r--')  # Perfect prediction line
    ax2.set_xlabel('True Angle (degrees)')
    ax2.set_ylabel('Predicted Angle (degrees)')
    ax2.set_title('Angle Prediction Performance')
    
    plt.tight_layout()
    plt.savefig(f'regression_performance_{timestamp}.png')
    plt.close()

def main():
    print("CSI Localization Model Training")
    print("=" * 50)
    
    # Load dataset
    X_train, X_test, y_train, y_test = load_dataset()
    print(f"Dataset loaded: {X_train.shape[0]} training samples, {X_test.shape[0]} testing samples")
    print(f"Feature dimension: {X_train.shape[1]}")
    
    # Prepare data
    (X_train_scaled, X_test_scaled,
     y_train_class, y_test_class,
     y_train_reg, y_test_reg) = prepare_data(X_train, X_test, y_train, y_test)
    
    # Train and evaluate classification models
    class_results = train_classification_models(
        X_train_scaled, X_test_scaled,
        y_train_class, y_test_class
    )
    
    # Train and evaluate regression models
    reg_results = train_regression_model(
        X_train_scaled, X_test_scaled,
        y_train_reg, y_test_reg
    )
    
    # Generate visualizations
    print("\nGenerating performance visualizations...")
    plot_results(class_results, reg_results, y_test_class, y_test_reg)
    print("\nTraining complete. Check the generated visualization files for detailed performance analysis.")

if __name__ == "__main__":
    main()
