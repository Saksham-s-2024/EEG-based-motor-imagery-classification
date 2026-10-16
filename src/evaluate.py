import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader

from src.data_loader import load_subject_raw
from src.preprocessing import preprocess_raw, extract_epochs, EEGDataset
from src.models.eegnet import EEGNet
from src.models.classical import build_csp_lda_pipeline


def evaluate_eegnet(model, dataloader, device):
    model.eval()
    all_preds = []
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for batch_X, batch_y in dataloader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_targets.extend(batch_y.numpy())

    return np.array(all_targets), np.array(all_preds), np.array(all_probs)


def plot_confusion_matrices(y_true, eegnet_preds, csp_preds):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # EEGNet Confusion Matrix
    cm_eegnet = confusion_matrix(y_true, eegnet_preds)
    sns.heatmap(cm_eegnet, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['Left Hand', 'Right Hand'], yticklabels=['Left Hand', 'Right Hand'])
    axes[0].set_title('EEGNet Confusion Matrix')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True Label')

    # CSP + LDA Confusion Matrix
    cm_csp = confusion_matrix(y_true, csp_preds)
    sns.heatmap(cm_csp, annot=True, fmt='d', cmap='Greens', ax=axes[1],
                xticklabels=['Left Hand', 'Right Hand'], yticklabels=['Left Hand', 'Right Hand'])
    axes[1].set_title('CSP + LDA Confusion Matrix')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('True Label')

    plt.tight_layout()
    plt.savefig('confusion_matrices.png')
    print("Saved confusion matrices plot to 'confusion_matrices.png'.")


if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # 1. Evaluate on unseen Holdout Subject (Subject 5)
    val_subject = 5
    raw_val = load_subject_raw(subject_id=val_subject)
    raw_val_clean = preprocess_raw(raw_val)
    _, X_val, y_val = extract_epochs(raw_val_clean)

    # Load trained EEGNet
    eegnet = EEGNet(channels=64, samples=480, num_classes=2)
    eegnet.load_state_dict(torch.load("eegnet_checkpoint.pth"))
    eegnet.to(device)

    val_dataset = EEGDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    y_true, eegnet_preds, eegnet_probs = evaluate_eegnet(eegnet, val_loader, device)

    # 2. Fit and Evaluate CSP + LDA Baseline on Subject 1-4 train data vs Subject 5
    print("\nTraining CSP + LDA baseline across Subjects 1-4...")
    train_X_list, train_y_list = [], []
    for sub in [1, 2, 3, 4]:
        raw = load_subject_raw(subject_id=sub)
        raw_clean = preprocess_raw(raw)
        _, X, y = extract_epochs(raw_clean)
        train_X_list.append(X)
        train_y_list.append(y)

    X_train_multi = np.concatenate(train_X_list, axis=0)
    y_train_multi = np.concatenate(train_y_list, axis=0)

    csp_lda = build_csp_lda_pipeline()
    csp_lda.fit(X_train_multi, y_train_multi)
    csp_preds = csp_lda.predict(X_val)

    # 3. Print Evaluation Reports
    print("\n================ EEGNET EVALUATION REPORT (Subject 5) ================")
    print(classification_report(y_true, eegnet_preds, target_names=['Left Hand', 'Right Hand']))
    print(f"ROC-AUC Score: {roc_auc_score(y_true, eegnet_probs):.4f}")

    print("\n================ CSP + LDA EVALUATION REPORT (Subject 5) ================")
    print(classification_report(y_true, csp_preds, target_names=['Left Hand', 'Right Hand']))

    # Plot & Save
    plot_confusion_matrices(y_true, eegnet_preds, csp_preds)