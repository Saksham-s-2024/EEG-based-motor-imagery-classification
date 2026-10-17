import requests
import numpy as np

# Create mock EEG trial tensor matching input dimensions (64 channels, 480 time samples)
dummy_signal = np.random.randn(64, 480).tolist()

payload = {
    "data": dummy_signal
}

url = "http://127.0.0.1:8000/predict"

print("Sending inference request to FastAPI ONNX Server...")
response = requests.post(url, json=payload)

if response.status_code == 200:
    print("\nInference Output Received:")
    print(response.json())
else:
    print(f"Error {response.status_code}:", response.text)