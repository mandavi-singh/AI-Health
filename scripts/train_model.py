import os
import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

#  Config
DATASET_PKL  = 'Dataset/breathing_dataset.pkl'
MODELS_DIR   = 'models'
BATCH_SIZE   = 64
EPOCHS       = 20
LR           = 1e-3
os.makedirs(MODELS_DIR, exist_ok=True)

# Label Encoding 
# Keep only main 3 classes, merge rare ones into Normal
LABEL_MAP = {
    'Normal':            0,
    'Hypopnea':          1,
    'Obstructive Apnea': 2,
    'Body event':        0,   # treat as Normal
    'Mixed Apnea':       2,   # treat as Obstructive Apnea
}
CLASS_NAMES = ['Normal', 'Hypopnea', 'Obstructive Apnea']

#  Load Dataset 
print("Loading dataset...")
df = pd.read_pickle(DATASET_PKL)
df['label_enc'] = df['label'].map(LABEL_MAP)
df = df.dropna(subset=['label_enc'])
df['label_enc'] = df['label_enc'].astype(int)
print(f"Total windows: {len(df)}")
print(f"Label distribution:\n{df['label'].value_counts()}\n")

participants = sorted(df['participant'].unique())
print(f"Participants: {participants}\n")

# Dataset Class 
class SleepDataset(Dataset):
    def __init__(self, df):
        # Stack flow + thorac + spo2 as 3-channel input
        # flow & thorac: 960 samples, spo2: 120 samples → upsample spo2 to 960
        flows   = np.array(df['flow_window'].tolist(),   dtype=np.float32)    # (N, 960)
        thoracs = np.array(df['thorac_window'].tolist(), dtype=np.float32)    # (N, 960)
        spo2s   = np.array(df['spo2_window'].tolist(),   dtype=np.float32)    # (N, 120)

        # Normalize each channel to zero mean, unit std
        flows   = (flows   - flows.mean(axis=1, keepdims=True))   / (flows.std(axis=1,   keepdims=True) + 1e-8)
        thoracs = (thoracs - thoracs.mean(axis=1, keepdims=True)) / (thoracs.std(axis=1, keepdims=True) + 1e-8)
        spo2s   = (spo2s   - spo2s.mean(axis=1, keepdims=True))   / (spo2s.std(axis=1,  keepdims=True) + 1e-8)

        # Upsample spo2 from 120 → 960 using repeat
        spo2s_up = np.repeat(spo2s, 8, axis=1)   # 120 * 8 = 960

        # Shape: (N, 3, 960)
        self.X = np.stack([flows, thoracs, spo2s_up], axis=1)
        self.y = np.array(df['label_enc'].tolist(), dtype=np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])

# 1D CNN Model 
class CNN1D(nn.Module):
    def __init__(self, n_classes=3):
        super(CNN1D, self).__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv1d(in_channels=3, out_channels=32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),          # 960 → 480

            # Block 2
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),          # 480 → 240

            # Block 3
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),          # 240 → 120

            # Block 4
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),              # → 128 x 1
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, n_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

#  Training Function 
def train_model(train_df, device):
    dataset    = SleepDataset(train_df)
    loader     = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model      = CNN1D(n_classes=3).to(device)

    # Class weights to handle imbalance
    label_counts = train_df['label_enc'].value_counts().sort_index()
    total        = len(train_df)
    weights      = torch.tensor([total / (3 * label_counts.get(i, 1)) for i in range(3)],
                                dtype=torch.float32).to(device)
    criterion  = nn.CrossEntropyLoss(weight=weights)
    optimizer  = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler  = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            out  = model(X_batch)
            loss = criterion(out, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        if (epoch + 1) % 5 == 0:
            avg_loss = total_loss / len(loader)
            print(f"    Epoch [{epoch+1}/{EPOCHS}]  Loss: {avg_loss:.4f}")

    return model

#  Evaluation Function 
def evaluate_model(model, test_df, device):
    dataset = SleepDataset(test_df)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            out     = model(X_batch)
            preds   = torch.argmax(out, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y_batch.numpy())

    return np.array(all_labels), np.array(all_preds)

#  Leave-One-Participant-Out Cross Validation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}\n")

fold_results = []
all_true, all_pred = [], []

for fold, test_pid in enumerate(participants):
    print(f"{'='*55}")
    print(f"Fold {fold+1}/{len(participants)}  |  Test: {test_pid}  |  Train: {[p for p in participants if p != test_pid]}")
    print(f"{'='*55}")

    train_df = df[df['participant'] != test_pid].reset_index(drop=True)
    test_df  = df[df['participant'] == test_pid].reset_index(drop=True)

    print(f"  Train: {len(train_df)} windows  |  Test: {len(test_df)} windows")

    model  = train_model(train_df, device)
    y_true, y_pred = evaluate_model(model, test_df, device)

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec  = recall_score(y_true, y_pred, average='macro', zero_division=0)
    cm   = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    fold_results.append({'fold': fold+1, 'test_pid': test_pid,
                          'accuracy': acc, 'precision': prec, 'recall': rec})
    all_true.extend(y_true)
    all_pred.extend(y_pred)

    # Save model
    model_path = os.path.join(MODELS_DIR, f'cnn_model_fold{fold+1}_{test_pid}.pt')
    torch.save(model.state_dict(), model_path)

    print(f"\n  📊 Results for Fold {fold+1} (Test: {test_pid})")
    print(f"     Accuracy : {acc:.4f}")
    print(f"     Precision: {prec:.4f}  (macro)")
    print(f"     Recall   : {rec:.4f}  (macro)")
    print(f"\n  Confusion Matrix (rows=True, cols=Predicted):")
    print(f"  Classes: {CLASS_NAMES}")
    print(f"  {cm}\n")

#  Overall Results
print(f"\n{'='*55}")
print("OVERALL RESULTS (All Folds)")
print(f"{'='*55}")

results_df = pd.DataFrame(fold_results)
print(results_df.to_string(index=False))

print(f"\n  Mean Accuracy : {results_df['accuracy'].mean():.4f} ± {results_df['accuracy'].std():.4f}")
print(f"  Mean Precision: {results_df['precision'].mean():.4f} ± {results_df['precision'].std():.4f}")
print(f"  Mean Recall   : {results_df['recall'].mean():.4f} ± {results_df['recall'].std():.4f}")

overall_cm = confusion_matrix(all_true, all_pred, labels=[0, 1, 2])
print(f"\n  Overall Confusion Matrix:")
print(f"  Classes: {CLASS_NAMES}")
print(f"  {overall_cm}")

# Save results CSV
results_df.to_csv(os.path.join(MODELS_DIR, 'cv_results.csv'), index=False)
print(f"\n✅ Results saved to {MODELS_DIR}/cv_results.csv")
print("✅ All fold models saved in models/ folder")

