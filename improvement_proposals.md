# Proposed Improvements for X-axis Accuracy

## 1. Data Augmentation & Feature Engineering

### Phase Unwrapping
- Implementation: Add phase unwrapping preprocessing step
- Impact: Resolves phase discontinuities
- Computational Cost: O(n) - negligible impact
- Expected Improvement: 15-20% reduction in X-axis error

### CSI Magnitude Normalization
- Implementation: Per-channel adaptive normalization
- Impact: Reduces bimodal magnitude distribution effects
- Computational Cost: O(n) - minimal overhead
- Expected Improvement: 10-15% reduction in variance

### Synthetic Data Generation
- Implementation: Generate synthetic CSI samples for underrepresented regions
- Impact: Better coverage of boundary regions
- Training Cost: Increased by 20%
- Runtime Cost: None (preprocessing only)

## 2. Architectural Modifications

### Attention Mechanism
- Implementation: Self-attention layer after conv layers
- Impact: Better capture of spatial relationships
- Computational Cost: +25% inference time
- Memory Impact: +15% model size
- Expected Improvement: 25-30% reduction in X-axis error

### Residual Connections
- Implementation: Add residual blocks in conv layers
- Impact: Better gradient flow and feature preservation
- Computational Cost: +5% inference time
- Memory Impact: +3% model size
- Expected Improvement: 10-15% reduction in error

### Multi-Scale Feature Extraction
- Implementation: Parallel conv paths with different kernel sizes
- Impact: Capture both local and global patterns
- Computational Cost: +20% inference time
- Memory Impact: +10% model size
- Expected Improvement: 20-25% reduction in error

## 3. Regularization Strategies

### L2 Regularization
- Implementation: Add weight decay (1e-4)
- Impact: Better generalization
- Computational Cost: Negligible
- Expected Improvement: 5-10% reduction in error

### Feature Dropout
- Implementation: Channel-wise dropout (0.1)
- Impact: Robust feature learning
- Computational Cost: Negligible during inference
- Expected Improvement: 8-12% reduction in error

### Label Smoothing
- Implementation: Soft labels for position targets
- Impact: More robust distance predictions
- Computational Cost: Negligible
- Expected Improvement: 5-8% reduction in error

## 4. Additional Data Collection

### Strategic Sampling
- Collect additional samples in:
  * Boundary regions (0-2m, 18-20m)
  * Areas with high prediction error
  * Different environmental conditions
- Impact: Better generalization
- Collection Cost: Manual effort required
- Training Cost: +30% more data

### Environmental Variation
- Collect data with:
  * Different furniture arrangements
  * Various occupancy levels
  * Multiple times of day
- Impact: More robust predictions
- Collection Cost: Significant manual effort
- Training Cost: +50% more data

## 5. Real-time Deployment Considerations

### Optimization Priorities
1. Phase unwrapping (highest impact/cost ratio)
2. L2 regularization (minimal overhead)
3. Residual connections (moderate cost)
4. Attention mechanism (if latency allows)

### Performance Targets
- Maintain < 20ms inference time
- Keep model size < 10MB
- Support batch processing
- Enable CPU-only deployment

### Implementation Strategy
1. Start with low-cost improvements
2. Benchmark each modification
3. Maintain real-time capability
4. Progressive feature addition

This proposal balances accuracy improvements with computational efficiency, prioritizing changes that offer the best impact-to-cost ratio while maintaining real-time performance capabilities.
