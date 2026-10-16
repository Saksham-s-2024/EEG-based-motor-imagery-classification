import os
import mne
from mne.datasets import eegbci
from config.config import DATA_DIR, MOTOR_IMAGERY_RUNS


def load_subject_raw(subject_id: int, runs: list = MOTOR_IMAGERY_RUNS) -> mne.io.Raw:
    """
    Downloads and loads raw EDF files for a specified subject and set of runs.
    Concatenates multiple runs into a single continuous MNE Raw object and
    standardizes channel montages to the 10-20 coordinate system.
    """
    # Set MNE data directory
    mne.set_config('PHYSIONET_EEGBCI_PATH', DATA_DIR, set_env=True)
    
    # Download / locate EDF file paths from PhysioNet
    raw_fnames = eegbci.load_data(subjects=[subject_id], runs=runs, path=DATA_DIR, verbose=False)    
    # Load raw EDF files into memory
    raw_list = [mne.io.read_raw_edf(f, preload=True, verbose=False) for f in raw_fnames]
    
    # Concatenate individual run streams into a continuous Raw array
    raw = mne.concatenate_raws(raw_list)
    
    # Standardize electrode placement names to the standard 10-20 montage
    eegbci.standardize(raw)
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage, on_missing='ignore')
    
    return raw


if __name__ == "__main__":
    # Test ingestion for Subject 1
    test_subject = 1
    print(f"Fetching raw EEG data for Subject {test_subject}...")
    
    raw_data = load_subject_raw(subject_id=test_subject)
    
    print("\n================ DATA INGESTION SUMMARY ================")
    print(f"Subject ID:          {test_subject}")
    print(f"Sampling Frequency: {raw_data.info['sfreq']} Hz")
    print(f"Total Channels:     {len(raw_data.ch_names)}")
    print(f"Signal Duration:    {raw_data.times[-1]:.2f} seconds")
    print("========================================================\n")
    
    # Inspect embedded event annotations
    events, event_id = mne.events_from_annotations(raw_data, verbose=False)
    print("Detected Event Annotations:")
    for key, value in event_id.items():
        count = (events[:, 2] == value).sum()
        print(f"  Marker '{key}' (Code {value}): {count} occurrences")