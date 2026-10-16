import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
import numpy as np

from src.data_loader import load_subject_raw
from src.preprocessing import preprocess_raw, extract_epochs, EEGDataset
from src.models.eegnet import EEGNet


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps") # Apple Silicon GPU
    return torch.device("cpu")


def load_multi_subject_dataset(subject_ids: list[int]):
    """
    In-memory dataset loader across a group of subjects for Phase 3 training.
    """
    datasets = []
    print(f"Loading and preprocessing data for {len(subject_ids)} subjects...")
    for sub in subject_ids:
        raw = load_subject_raw(subject_id=sub)
        raw_clean = preprocess_raw(raw)
        _, X, y = extract_epochs(raw_clean)
        datasets.append(EEGDataset(X, y))
    return ConcatDataset(datasets)


def train_eegnet(model, train_loader, val_loader, epochs=25, lr=1e-3, device="cpu"):
    model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"\nBeginning EEGNet Training on {device.type.upper()}...")
    print("=" * 60)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_X.size(0)
            preds = outputs.argmax(dim=1)
            train_correct += (preds == batch_y).sum().item()
            train_total += batch_y.size(0)

        scheduler.step()
        train_acc = (train_correct / train_total) * 100.0

        # Validation Loop
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)

                val_loss += loss.item() * batch_X.size(0)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == batch_y).sum().item()
                val_total += batch_y.size(0)

        val_acc = (val_correct / val_total) * 100.0 if val_total > 0 else 0.0

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:02d}/{epochs:02d} | "
                  f"Train Loss: {train_loss/train_total:.4f} | Train Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss/val_total:.4f} | Val Acc: {val_acc:.2f}%")

    print("=" * 60)
    return model


if __name__ == "__main__":
    device = get_device()
    
    # Train on Subjects 1-4, Validate on Subject 5
    train_subjects = [1, 2, 3, 4]
    val_subjects = [5]

    train_dataset = load_multi_subject_dataset(train_subjects)
    val_dataset = load_multi_subject_dataset(val_subjects)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    model = EEGNet(channels=64, samples=480, num_classes=2)
    trained_model = train_eegnet(model, train_loader, val_loader, epochs=20, lr=1e-3, device=device)
    
    # Save checkpoint
    torch.save(trained_model.state_dict(), "eegnet_checkpoint.pth")
    print("Model checkpoint saved to 'eegnet_checkpoint.pth'.")