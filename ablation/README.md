# Ablation Study Scripts & Benchmarks
# Course: CSC8370 Data Security

This folder contains isolated, standalone ablation scripts to systematically evaluate individual components of our model and defense architectures:

1. `ablation_pooling.py`:
   - Evaluates **Max Pooling vs. Average Pooling** on the primary CNN architecture across 10 rounds of federated training.
2. `ablation_trimming.py`:
   - Evaluates **Coordinate-wise Trimmed Mean vs. Standard FedAvg (No Trimming)** after filtering Byzantine clients.
3. `ablation_detection.py`:
   - Evaluates **Individual Anomaly Detectors** (Cosine-only vs. L2-Norm-only vs. Multi-Krum-only vs. Full Multi-Metric Pipeline).
4. `ablation_trim_ratio.py`:
   - Evaluates sensitivity across different **Trimming Ratios** (beta = 0.0, 0.1, 0.2, 0.3).
