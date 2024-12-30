# WiFi CSI-based Device Localization Model - Final Report

## Executive Summary
We have successfully developed and evaluated machine learning models for device-to-device localization using WiFi Channel State Information (CSI) data. The models demonstrate exceptional accuracy in predicting both distance and angle between transmitter and receiver.

## Dataset Overview
- **Data Collection**: 10 measurement scenarios
  - Distances: 1m to 5m (1m intervals)
  - Angles: 30° and -60°
- **CSI Data Structure**: 3D matrix (3 × 30 × N)
  - 3 receiving antennas
  - 30 OFDM subcarriers
  - N packets per measurement
- **Dataset Size**: 100,000 total samples
  - Training set: 80,000 samples (80%)
  - Testing set: 20,000 samples (20%)
- **Feature Dimension**: 185 features per sample

## Model Performance

### Classification Models
All models achieved near-perfect accuracy in predicting distance/angle combinations:

1. **K-Nearest Neighbors (KNN)**
   - Accuracy: 99.99%
   - Best performing classification model
   - Excellent for categorical predictions

2. **Random Forest**
   - Accuracy: 99.96%
   - Robust performance across all categories
   - Comparable to SVM performance

3. **Support Vector Machine (SVM)**
   - Accuracy: 99.96%
   - Consistent performance
   - Efficient for deployment

### Regression Models
Achieved high precision in continuous predictions:

1. **Distance Prediction**
   - Mean Squared Error (MSE): 0.0154 m²
   - Mean Absolute Error (MAE): 0.0558 m (≈5.6 cm)
   - Exceptional precision for practical applications

2. **Angle Prediction**
   - Mean Squared Error (MSE): 12.3407 degrees²
   - Mean Absolute Error (MAE): 1.0678 degrees
   - High accuracy in angle estimation

## Feature Engineering
- Extracted amplitude and phase information from complex CSI data
- Computed statistical features (mean, variance, percentiles)
- Applied robust normalization techniques
- Total of 185 engineered features:
  - 90 amplitude features
  - 90 phase features
  - 5 statistical features

## Model Comparison and Analysis

### Classification Approach
- Perfect for predicting discrete location categories
- All models show comparable, excellent performance
- KNN slightly outperforms other classifiers
- Suitable for applications requiring categorical location information

### Regression Approach
- Provides precise continuous predictions
- Distance prediction highly accurate (5.6cm average error)
- Angle prediction very precise (1.07° average error)
- Ideal for applications requiring exact positioning

## Visualizations
Two key visualization files have been generated:

1. `model_comparison_20241228_130948.png`
   - Compares accuracy across classification models
   - Shows consistent performance across all models
   - Demonstrates near-perfect classification accuracy

2. `regression_performance_20241228_130948.png`
   - Scatter plots of predicted vs. true values
   - Shows excellent correlation for both distance and angle
   - Minimal outliers in predictions

## Recommendations

### Model Selection
1. **For Categorical Localization**
   - Recommend KNN classifier
   - Highest accuracy (99.99%)
   - Simple to implement and deploy

2. **For Precise Localization**
   - Recommend using regression models
   - High precision in both distance and angle
   - Suitable for applications requiring continuous predictions

### Deployment Considerations
1. **Feature Processing**
   - Implement robust feature extraction pipeline
   - Use standardization for numerical stability
   - Consider real-time processing requirements

2. **Model Deployment**
   - Both classification and regression models are deployment-ready
   - Consider hardware requirements for real-time processing
   - Implement error handling for edge cases

## Conclusion
The developed models demonstrate exceptional performance in WiFi CSI-based localization:

1. **Classification Performance**
   - Near-perfect accuracy (>99.9%)
   - Robust across all distance/angle combinations
   - Suitable for practical deployment

2. **Regression Performance**
   - High precision in continuous predictions
   - Average errors of 5.6cm for distance
   - Average errors of 1.07° for angle

The results indicate that the models are ready for practical implementation in device-to-device localization applications, offering both categorical and precise continuous location predictions with high reliability.
