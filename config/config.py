import os

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

# PhysioNet Dataset Config
# Runs 4, 8, 12 correspond to Motor Imagery: Imagined Left Hand (T1) vs Imagined Right Hand (T2)
MOTOR_IMAGERY_RUNS = [4, 8, 12]

# Target Sampling Rate (PhysioNet native rate is 160 Hz)
SAMPLING_RATE = 160

# Signal Filtering Frequencies
LOW_CUT = 8.0   # Hz (Mu rhythm lower bound)
HIGH_CUT = 30.0 # Hz (Beta rhythm upper bound)

# Classification Event Mapping
EVENT_ID = {
    'Rest': 1,      # T0
    'Left_Hand': 2, # T1
    'Right_Hand': 3 # T2
}