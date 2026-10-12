import mne
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from config.config import LOW_CUT, HIGH_CUT, SAMPLING_RATE
from src.data_loader import load_subject_raw


def preprocess_raw(raw: mne.io.Raw, l_freq: float = LOW_CUT, h_freq: float = HIGH_CUT) -> mne.io.Raw:
    """
    Applies zero-phase bandpass filtering (8-30 Hz) and notch filtering (50/60 Hz)
    to isolate Mu and Beta sensorimotor rhythms while eliminating grid noise.
    """
    raw_filtered = raw.copy()
    
    # 1. Bandpass Filter (8 - 30 Hz)
    raw_filtered.filter(l_freq=l_freq, h_freq=h_freq, fir_design='firwin', verbose=False)
    
    # 2. Notch Filter (50 Hz and 60 Hz)
    raw_filtered.notch_filter(freqs=[50.0, 60.0], verbose=False)
    
    return raw_filtered


def extract_epochs(raw: mne.io.Raw, tmin: float = 0.0, tmax: float = 3.0) -> tuple[mne.Epochs, np.ndarray, np.ndarray]:
    """
    Extracts 3-second motor imagery epochs around cue onset markers (T1 and T2).
    Maps T1 (Left Hand) -> Class 0, T2 (Right Hand) -> Class 1.
    """
    # Extract event array from annotations
    events, event_dict = mne.events_from_annotations(raw, verbose=False)
    
    # Identify annotation keys corresponding to motor imagery (T1 and T2)
    # T1 = Left Hand, T2 = Right Hand
    target_event_id = {}
    for key, val in event_dict.items():
        if 'T1' in key:
            target_event_id['Left_Hand'] = val
        elif 'T2' in key:
            target_event_id['Right_Hand'] = val

    if not target_event_id:
        raise ValueError("No T1 or T2 motor imagery events found in annotations.")

    # Slice epochs from tmin (0s) to tmax (3s) post-cue
    epochs = mne.Epochs(
        raw,
        events,
        event_id=target_event_id,
        tmin=tmin,
        tmax=tmax - (1.0 / SAMPLING_RATE), # Exactly 480 samples at 160 Hz
        baseline=None,
        preload=True,
        verbose=False
    )
    
    # Convert event labels to 0-indexed binary targets (0: Left Hand, 1: Right Hand)
    # MNE events use integer event codes in column 2
    raw_labels = epochs.events[:, 2]
    unique_ids = sorted(list(target_event_id.values()))
    label_map = {unique_ids[0]: 0, unique_ids[1]: 1}
    
    y = np.array([label_map[code] for code in raw_labels])
    X = epochs.get_data() # Array shape: (n_epochs, n_channels, n_times)
    
    return epochs, X, y


class EEGDataset(Dataset):
    """
    PyTorch Dataset wrapper converting 3D numpy arrays (Epochs, Channels, Times)
    into 4D Tensors (Epochs, 1, Channels, Times) for EEGNet.
    """
    def __init__(self, X: np.ndarray, y: np.ndarray):
        # Add single channel dimension: (B, C, T) -> (B, 1, C, T)
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


if __name__ == "__main__":
    # Test Phase 2 Preprocessing & Epoching pipeline on Subject 1
    test_subject = 1
    print(f"Executing Preprocessing & Epoch Extraction for Subject {test_subject}...")
    
    raw = load_subject_raw(subject_id=test_subject)
    raw_clean = preprocess_raw(raw)
    epochs, X, y = extract_epochs(raw_clean)
    
    # Wrap in PyTorch Dataset & DataLoader
    dataset = EEGDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # Inspect shape
    sample_batch_X, sample_batch_y = next(iter(dataloader))
    
    print("\n================ PREPROCESSING & EPOCH SUMMARY ================")
    print(f"Total Epochs Extracted:  {len(y)}")
    print(f"Class Distribution:      Class 0 (Left Hand): {(y == 0).sum()} | Class 1 (Right Hand): {(y == 1).sum()}")
    print(f"Numpy Array Shape (X):   {X.shape}  -> (Epochs, Channels, Times)")
    print(f"PyTorch Batch Shape:    {sample_batch_X.shape} -> (Batch, 1, Channels, Times)")
    print(f"PyTorch Label Shape:    {sample_batch_y.shape}")
    print("===============================================================\n")