import torch

from ai_models.hazard_traversability.dataset_utils import prepare_numpy_dataset
from ai_models.hazard_traversability.model import SimpleHazardCNN, logits_to_outputs


def run_inference_demo():
    X, y = prepare_numpy_dataset(split="train", num_samples=8, raw_limit=100)

    X_t = torch.tensor(X).permute(0, 3, 1, 2)

    model = SimpleHazardCNN(num_classes=3)
    model.eval()

    with torch.no_grad():
        logits = model(X_t)
        outputs = logits_to_outputs(logits)

    print("=== HAZARD MODEL INFERENCE DEMO ===")
    for i, out in enumerate(outputs):
        print(f"Sample {i}: true_label={y[i]}, output={out}")


if __name__ == "__main__":
    run_inference_demo()
