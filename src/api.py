from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import onnxruntime as ort
import numpy as np

app = FastAPI(title="EEG Motor Imagery Classification API")

# Load ONNX Session
session = ort.InferenceSession("eegnet.onnx", providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name


class EEGInferenceRequest(BaseModel):
    # Expects a 3D list matching shape: (64 channels, 480 time samples)
    data: list[list[float]]


@app.get("/")
def health_check():
    return {"status": "online", "model": "EEGNet ONNX"}


@app.post("/predict")
def predict(request: EEGInferenceRequest):
    arr = np.array(request.data, dtype=np.float32)

    if arr.shape != (64, 480):
        raise HTTPException(
            status_code=400,
            detail=f"Expected EEG signal shape (64, 480), got {arr.shape}"
        )

    # Format into 4D tensor shape: (1, 1, 64, 480)
    tensor_input = np.expand_dims(np.expand_dims(arr, axis=0), axis=0)

    outputs = session.run(None, {input_name: tensor_input})
    logits = outputs[0][0]

    # Compute Softmax probabilities
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / exp_logits.sum()

    predicted_class = int(np.argmax(probs))
    class_map = {0: "Left Hand", 1: "Right Hand"}

    return {
        "prediction": class_map[predicted_class],
        "class_id": predicted_class,
        "confidence": float(probs[predicted_class]),
        "probabilities": {
            "Left Hand": float(probs[0]),
            "Right Hand": float(probs[1])
        }
    }