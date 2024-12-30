# WiFi CSI Device Localization

This repository contains a machine learning model for device-to-device localization using WiFi Channel State Information (CSI) data.

## Project Structure

- `data/`: Contains CSI measurement files (.mat format)
- `*.py`: Python scripts for data processing and model training
- `final_report.md`: Detailed analysis and results
- `readme.txt`: Original project requirements

## Setup Instructions

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install numpy scipy pandas scikit-learn matplotlib
   ```
3. Run the scripts in order:
   - verify_mat.py
   - load_csi_data.py
   - preprocess_csi.py
   - split_dataset.py
   - train_models.py

## Data Description

The dataset includes CSI measurements at:
- Distances: 1m to 5m
- Angles: 30° and -60°
- Format: 3D matrix (3 x 30 x number of packets)
  - 3 receiving antennas
  - 30 OFDM subcarriers
  - Variable number of packets
