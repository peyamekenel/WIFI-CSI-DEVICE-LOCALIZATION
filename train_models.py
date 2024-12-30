import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, mean_squared_error, mean_absolute_error,
                           confusion_matrix, classification_report)
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings

def load_dataset():
    """Load the preprocessed and split dataset."""
    print("Loading dataset...")
    with open('csi_dataset.pkl', 'rb') as f:
        dataset = pickle.load(f)
    return dataset['X_train'], dataset['X_test'], dataset['y_train'], dataset['y_test']

def prepare_data(X_train, X_test, y_train, y_test):
    """Prepare data for training by creating classification labels."""
    # Features are already normalized in split_dataset.py
    
    # Prepare classification labels (combine distance and angle into categories)
    y_train_class = np.array([f"{d}m_{a}deg" for d, a in y_train])
    y_test_class = np.array([f"{d}m_{a}deg" for d, a in y_test])
    
    # Prepare regression targets (separate distance and angle)
    y_train_reg = y_train
    y_test_reg = y_test
    
    return (X_train, X_test,  # Use original features, already normalized
            y_train_class, y_test_class,
            y_train_reg, y_test_reg)

def train_classification_models(X_train, X_test, y_train, y_test):
    """Train and evaluate classification models with nested cross-validation."""
    # Print dataset statistics
    print("\nDataset Statistics:")
    print(f"Training set shape: {X_train.shape}")
    print(f"Testing set shape: {X_test.shape}")
    print(f"Number of unique training samples: {len(np.unique(X_train, axis=0))}")
    print(f"Number of unique testing samples: {len(np.unique(X_test, axis=0))}")
    print("\nFeature Statistics:")
    print(f"Training set - Mean: {np.mean(X_train):.4f}, Std: {np.std(X_train):.4f}")
    print(f"Testing set - Mean: {np.mean(X_test):.4f}, Std: {np.std(X_test):.4f}")
    print(f"Training labels - Unique values: {np.unique(y_train)}")
    print(f"Testing labels - Unique values: {np.unique(y_test)}")
    
    # Check for potential data leakage
    train_hashes = {hash(tuple(row)) for row in X_train}
    test_hashes = {hash(tuple(row)) for row in X_test}
    overlap = len(train_hashes.intersection(test_hashes))
    if overlap > 0:
        print(f"\nWARNING: Found {overlap} identical samples in both train and test sets!")

    # Initialize nested cross-validation
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    # Initialize arrays to store nested CV results
    n_samples = X_train.shape[0]
    outer_fold_predictions = np.zeros((n_samples,))
    outer_fold_probabilities = np.zeros((n_samples, len(np.unique(y_train))))
    
    print("\nPerforming nested cross-validation...")
    
    # Outer loop for unbiased performance estimation
    for fold_idx, (train_idx, val_idx) in enumerate(outer_cv.split(X_train, y_train), 1):
        X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
        y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
        
        # Feature selection within the fold
        n_features = min(25, X_train.shape[1])
        selector = SelectKBest(score_func=f_classif, k=n_features)
        X_fold_train_selected = selector.fit_transform(X_fold_train, y_fold_train)
        X_fold_val_selected = selector.transform(X_fold_val)
        
        # Add controlled noise for regularization (1% of std dev)
        noise_level = 0.01 * np.std(X_fold_train_selected)
        X_fold_train_selected += np.random.normal(0, noise_level, X_fold_train_selected.shape)
        
        print(f"\nOuter Fold {fold_idx}:")
        print(f"Selected {n_features} features")
    
    # Initialize models with stronger regularization
    # Define model configurations
    models = {
        'Random Forest': lambda: RandomForestClassifier(
            n_estimators=50,
            max_depth=3,
            min_samples_split=10,
            min_samples_leaf=8,
            max_features='sqrt',
            bootstrap=True,
            class_weight='balanced',
            random_state=42
        ),
        'SVM': lambda: SVC(
            kernel='rbf',
            C=0.01,
            gamma='auto',
            class_weight='balanced',
            probability=True,
            random_state=42,
            max_iter=2000
        ),
        'KNN': lambda: KNeighborsClassifier(
            n_neighbors=15,
            weights='distance',
            metric='euclidean',
            p=2,
            leaf_size=5
        )
    }
    
    results = {}
    print("\nTraining Classification Models:")
    print("-" * 50)
    
    for name, model_factory in models.items():
        print(f"\nTraining {name}...")
        model_cv_scores = []
        model_predictions = []
        
        # Outer loop predictions
        outer_predictions = np.zeros_like(y_train)
        
        # Perform nested cross-validation
        for fold_idx, (train_idx, val_idx) in enumerate(outer_cv.split(X_train, y_train), 1):
            X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
            y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
            
            # Feature selection within the fold
            selector = SelectKBest(score_func=f_classif, k=min(25, X_train.shape[1]))
            X_fold_train_selected = selector.fit_transform(X_fold_train, y_fold_train)
            X_fold_val_selected = selector.transform(X_fold_val)
            
            # Inner loop for hyperparameter tuning (if needed)
            best_score = 0
            best_model = None
            
            for _ in range(3):  # Try 3 different random states
                model = model_factory()
                inner_scores = []
                
                for inner_train_idx, inner_val_idx in inner_cv.split(X_fold_train_selected, y_fold_train):
                    X_inner_train = X_fold_train_selected[inner_train_idx]
                    X_inner_val = X_fold_train_selected[inner_val_idx]
                    y_inner_train = y_fold_train[inner_train_idx]
                    y_inner_val = y_fold_train[inner_val_idx]
                    
                    model.fit(X_inner_train, y_inner_train)
                    score = model.score(X_inner_val, y_inner_val)
                    inner_scores.append(score)
                
                avg_score = np.mean(inner_scores)
                if avg_score > best_score:
                    best_score = avg_score
                    best_model = model
            
            # Train best model on full fold training data
            best_model.fit(X_fold_train_selected, y_fold_train)
            fold_predictions = best_model.predict(X_fold_val_selected)
            outer_predictions[val_idx] = fold_predictions
            
            # Calculate fold score
            fold_score = accuracy_score(y_fold_val, fold_predictions)
            model_cv_scores.append(fold_score)
            print(f"Fold {fold_idx} accuracy: {fold_score:.4f}")
        
        # Final evaluation on test set
        selector_final = SelectKBest(score_func=f_classif, k=min(25, X_train.shape[1]))
        X_train_selected_final = selector_final.fit_transform(X_train, y_train)
        X_test_selected_final = selector_final.transform(X_test)
        
        final_model = model_factory()
        final_model.fit(X_train_selected_final, y_train)
        test_predictions = final_model.predict(X_test_selected_final)
        
        # Calculate final metrics
        cv_accuracy = accuracy_score(y_train, outer_predictions)
        test_accuracy = accuracy_score(y_test, test_predictions)
        cm = confusion_matrix(y_test, test_predictions)
        report = classification_report(y_test, test_predictions)
        
        results[name] = {
            'cv_accuracy': cv_accuracy,
            'test_accuracy': test_accuracy,
            'cv_predictions': outer_predictions,
            'test_predictions': test_predictions,
            'confusion_matrix': cm,
            'classification_report': report,
            'cv_scores': model_cv_scores
        }
        
        print(f"\n{name} Results:")
        print(f"Nested CV Accuracy: {cv_accuracy:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print("\nClassification Report:")
        print(report)
        
        # Print feature importance for Random Forest
        if name == 'Random Forest' and hasattr(final_model, 'feature_importances_'):
            importances = final_model.feature_importances_
            indices = np.argsort(importances)[::-1]
            print("\nTop 10 Most Important Features:")
            for f in range(min(10, X_train_selected_final.shape[1])):
                print(f"{f+1}. Feature {indices[f]}: {importances[indices[f]]:.4f}")
    
    return results

def train_regression_model(X_train, X_test, y_train, y_test):
    """Train and evaluate regression model for distance and angle prediction with feature selection."""
    print("\nTraining Regression Model:")
    print("-" * 50)
    
    # Perform feature selection for regression
    print("\nPerforming feature selection for regression...")
    distance_selector = SelectKBest(score_func=f_classif, k=min(100, X_train.shape[1]))
    angle_selector = SelectKBest(score_func=f_classif, k=min(100, X_train.shape[1]))
    
    # Select features for distance prediction
    X_train_distance = distance_selector.fit_transform(X_train, y_train[:, 0])
    X_test_distance = distance_selector.transform(X_test)
    
    # Select features for angle prediction
    X_train_angle = angle_selector.fit_transform(X_train, y_train[:, 1])
    X_test_angle = angle_selector.transform(X_test)
    
    print(f"Selected {X_train_distance.shape[1]} features for distance prediction")
    print(f"Selected {X_train_angle.shape[1]} features for angle prediction")
    
    # Train separate models for distance and angle with regularization
    distance_model = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=5,  # Limit tree depth
        min_samples_split=5,
        min_samples_leaf=4,
        learning_rate=0.1,
        subsample=0.8,  # Use 80% of samples for each tree
        random_state=42
    )
    angle_model = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=5,
        min_samples_split=5,
        min_samples_leaf=4,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42
    )
    
    # Train distance model
    print("Training distance predictor...")
    distance_model.fit(X_train_distance, y_train[:, 0])  # First column is distance
    distance_pred = distance_model.predict(X_test_distance)
    distance_mse = mean_squared_error(y_test[:, 0], distance_pred)
    distance_mae = mean_absolute_error(y_test[:, 0], distance_pred)
    
    # Train angle model
    print("Training angle predictor...")
    angle_model.fit(X_train_angle, y_train[:, 1])  # Second column is angle
    angle_pred = angle_model.predict(X_test_angle)
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
    
    # Plot classification results with cross-validation
    plt.figure(figsize=(10, 6))
    data = []
    labels = []
    for name, results in class_results.items():
        data.append(results['cv_scores'])
        labels.extend([name] * len(results['cv_scores']))
    plt.boxplot([results['cv_scores'] for results in class_results.values()],
                labels=class_results.keys())
    plt.title('Classification Model Performance (Cross-Validation)')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'model_comparison_cv_{timestamp}.png')
    plt.close()
    
    # Plot confusion matrices
    for name, results in class_results.items():
        plt.figure(figsize=(10, 8))
        sns.heatmap(results['confusion_matrix'], annot=True, fmt='d',
                   xticklabels=np.unique(y_test_class),
                   yticklabels=np.unique(y_test_class))
        plt.title(f'{name} Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.savefig(f'confusion_matrix_{name.lower().replace(" ", "_")}_{timestamp}.png')
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
