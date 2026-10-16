import torch
from src.models.eegnet import EEGNet


def export_to_onnx(model_path="eegnet_checkpoint.pth", output_path="eegnet.onnx"):
    model = EEGNet(channels=64, samples=480, num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    dummy_input = torch.randn(1, 1, 64, 480)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['eeg_signal'],
        output_names=['logits'],
        dynamic_axes={
            'eeg_signal': {0: 'batch_size'},
            'logits': {0: 'batch_size'}
        }
    )
    print(f"Model exported successfully to '{output_path}'.")


if __name__ == "__main__":
    export_to_onnx()