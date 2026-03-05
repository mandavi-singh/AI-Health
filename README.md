# AI for Health — Sleep Apnea Detection (SRIP 2026)

## Overview
This project detects breathing irregularities (Hypopnea, Obstructive Apnea) during sleep using physiological signals from overnight PSG recordings.

## Dataset
5 participants (AP01–AP05), each with 8-hour sleep recordings:
- Nasal Airflow (32 Hz)
- Thoracic Movement (32 Hz)
- SpO₂ / Oxygen Saturation (4 Hz)
- Flow Events (breathing event annotations)
- Sleep Profile (sleep stage annotations)

## Project Structure
```
Project Root/
├── Data/
│   ├── AP01/
│   │   ├── Flow - 30-05-2024.txt
│   │   ├── Thorac - 30-05-2024.txt
│   │   ├── SPO2 - 30-05-2024.txt
│   │   ├── Flow Events - 30-05-2024.txt
│   │   └── Sleep profile - 30-05-2024.txt
│   └── ...
├── Dataset/
│   ├── breathing_dataset.csv
│   └── breathing_dataset.pkl
├── Visualizations/
│   └── AP01_visualization.pdf
├── models/
│   ├── cnn_model_fold1_AP01.pt
│   └── cv_results.csv
├── scripts/
│   ├── vis.py
│   ├── create_dataset.py
│   └── train_model.py
├── README.md
└── requirements.txt
```

## Installation
```bash
pip install -r requirements.txt
```

## Usage

### 1. Visualization
```bash
python scripts/vis.py -name "Data/AP01"
```
Generates a PDF with all 3 signals and breathing events overlaid.

### 2. Dataset Creation
```bash
python scripts/create_dataset.py -in_dir "Data" -out_dir "Dataset"
```
Applies bandpass filter (0.17–0.4 Hz), creates 30s windows with 50% overlap, assigns labels.

### 3. Model Training
```bash
python scripts/train_model.py
```
Trains a 1D CNN using Leave-One-Participant-Out Cross Validation.

## Results

| Fold | Test Participant | Accuracy | Precision | Recall |
|------|-----------------|----------|-----------|--------|
| 1    | AP01            | 73.4%    | 0.374     | 0.553  |
| 2    | AP02            | 34.8%    | 0.358     | 0.577  |
| 3    | AP03            | 54.1%    | 0.332     | 0.265  |
| 4    | AP04            | 83.6%    | 0.392     | 0.411  |
| 5    | AP05            | 61.9%    | 0.429     | 0.524  |
| **Mean** | —           | **61.6%** | **0.377** | **0.466** |

## Dependencies
- Python 3.x
- pandas, numpy, scipy, matplotlib
- torch, scikit-learn

## Note
AI tools were used in the development of this project and are explicitly acknowledged as per submission guidelines.
