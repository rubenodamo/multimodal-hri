# Analysis

## Overview

This section describes how experimental data is analysed to evaluate the performance and usability of the multimodal HRI system.

Data is collected during each trial and stored as structured logs. These logs are processed using a dedicated analysis pipeline to compute quantitative metrics and generate visualisations.

The analysis focuses on comparing interaction modes (voice, gesture, multimodal) across performance, efficiency, and multimodal-specific behaviour.

## Data Collection

Each trial produces a structured log entry containing:

- Expected command (ground truth)
- Predicted command (system output)
- Interaction condition (voice, gesture, multimodal)
- Latency ($\text{ms}$)
- Correction count
- Confidence score
- Fusion metadata (multimodal only)

Logs are stored as CSV files and loaded into pandas DataFrames for analysis.

## Metrics

| Metric                | Description                                                      |
|-----------------------|------------------------------------------------------------------|
| **Accuracy**          | Whether the predicted command matches the expected command       |
| **Latency**           | Time taken from prompt to final accepted command ($\text{ms}$)   |
| **Correction Count**  | Number of retries before correct submission                      |
| **Conflict Rate**     | Frequency of disagreement between voice and gesture inputs       |
| **Fusion Timing**     | Whether inputs occurred within the fusion time window            |
| **Confidence**        | Final confidence score of the predicted command                  |

## Analysis Methods

The analysis pipeline computes metrics across multiple dimensions to evaluate system performance and multimodal behaviour.

| Analysis Type             | Description                                                  |
|---------------------------|--------------------------------------------------------------|
| Condition Comparison      | Compare metrics across voice, gesture, and multimodal modes  |
| Field-Level Analysis      | Evaluate accuracy for action, object, and location           |
| Multimodal Analysis       | Measure conflicts, fusion success, and temporal alignment    |
| Error Analysis            | Identify which fields are incorrect                          |
| Learning Analysis         | Track performance changes over trial sequence                |


## Visualisation

The analysis pipeline generates plots to support interpretation of results.

| Plot                          | Type        | Purpose                               |
|-------------------------------|-------------|---------------------------------------|
| Accuracy by condition         | Bar chart   | Compare performance across modalities |
| Latency by condition          | Box plot    | Show distribution of response times   |
| Error rate by condition       | Bar chart   | Compare failure rates                 |
| Corrections by condition      | Bar chart   | Measure user effort                   |
| Field-level accuracy          | Bar chart   | Compare action/object/location errors |
| Confidence vs accuracy        | Box plot    | Assess reliability of confidence      |
| Temporal gap distribution     | Histogram   | Analyse timing between inputs         |
| Learning curve (accuracy)     | Line chart  | Detect performance improvement        |
| Learning curve (latency)      | Line chart  | Detect efficiency improvement         |

Plots are generated using matplotlib and can be displayed interactively or saved to disk.

## Running the Analysis

The analysis pipeline can be executed via command line:

```bash
python -m analysis.run_analysis

# Analyse a specific file
python -m analysis.run_analysis --file path/to/session.csv

# Analyse all logs in a directory
python -m analysis.run_analysis --dir logs/

# Save plots to disk
python -m analysis.run_analysis --save-plots

# Custom output directory
python -m analysis.run_analysis --output-dir my/plots/
```
---