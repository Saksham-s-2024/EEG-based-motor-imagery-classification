import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np

from src.data_loader import load_subject_raw
from src.preprocessing import preprocess_raw, extract_epochs, EEGDataset
from src.models.eegnet import EEGNet
from sklearn.metrics import accuracy_score


def fine_tune_subject(target_subject_id=5, calibration_ratio=0.2):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Load target subject data
    raw = load_subject_raw(subject_id=target_subject_id)
    raw_clean = preprocess_raw(raw)
    _, X, y = extract_epochs(raw_clean)
    
    dataset = EEGDataset(X, y)
    
    # Split into calibration (train) and test sets
    calib_size = int(len(dataset) * calibration_ratio)
    test_size = len(dataset) - calib_size
    calib_dataset, test_dataset = random_split(
        dataset, [calib_size, test_size], generator=torch.Generator().manual_seed(42)
    )

    calib_loader = DataLoader(calib_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Load base pre-trained EEGNet model
    model = EEGNet(channels=64, samples=480, num_classes=2)
    model.load_state_dict(torch.load("eegnet_checkpoint.pth"))
    model.to(device)

    # Freeze earlier spatial/temporal feature layers
    for param in model.conv1.parameters():
        param.requires_grad = False
    for param in model.batchnorm1.parameters():
        param.requires_grad = False
    if hasattr(model, 'depthwiseConv'):
        for param in model.depthwiseConv.parameters():
            param.requires_grad = False

    # Fine-tune with a low learning rate
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    print(f"\nFine-tuning EEGNet on Subject {target_subject_id} using {calib_size} calibration trials...")
    model.train()
    for epoch in range(1, 16):
        for batch_X, batch_y in calib_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    # Evaluate on remaining holdout test trials for this subject
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(batch_y.numpy())

    acc = accuracy_score(all_targets, all_preds) * 100.0
    print(f"Fine-Tuned EEGNet Accuracy on Subject {target_subject_id} (Test Set): {acc:.2f}%")


if __name__ == "__main__":
    fine_tune_subject(target_subject_id=5, calibration_ratio=0.2)