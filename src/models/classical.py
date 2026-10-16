import numpy as np
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline


def build_csp_lda_pipeline(n_components: int = 4) -> Pipeline:
    """
    Builds a CSP + LDA classification pipeline.
    CSP maximizes variance for one class while minimizing it for another.
    """
    csp = CSP(n_components=n_components, log=True, norm_trace=False)
    lda = LinearDiscriminantAnalysis()
    return Pipeline([('CSP', csp), ('LDA', lda)])


def build_csp_svm_pipeline(n_components: int = 4) -> Pipeline:
    """
    Builds a CSP + SVM classification pipeline.
    """
    csp = CSP(n_components=n_components, log=True, norm_trace=False)
    svm = SVC(kernel='rbf', C=1.0, probability=True)
    return Pipeline([('CSP', csp), ('SVM', svm)])


if __name__ == "__main__":
    from src.data_loader import load_subject_raw
    from src.preprocessing import preprocess_raw, extract_epochs
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    # Load and preprocess sample data
    raw = load_subject_raw(subject_id=1)
    raw_clean = preprocess_raw(raw)
    _, X, y = extract_epochs(raw_clean)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Test CSP + LDA
    lda_pipeline = build_csp_lda_pipeline()
    lda_pipeline.fit(X_train, y_train)
    lda_preds = lda_pipeline.predict(X_test)
    print(f"CSP + LDA Baseline Accuracy (Subject 1 Holdout): {accuracy_score(y_test, lda_preds) * 100:.2f}%")

    # Test CSP + SVM
    svm_pipeline = build_csp_svm_pipeline()
    svm_pipeline.fit(X_train, y_train)
    svm_preds = svm_pipeline.predict(X_test)
    print(f"CSP + SVM Baseline Accuracy (Subject 1 Holdout): {accuracy_score(y_test, svm_preds) * 100:.2f}%")