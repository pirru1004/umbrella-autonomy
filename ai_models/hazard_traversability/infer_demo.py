import torch
import matplotlib.pyplot as plt

from ai_models.hazard_traversability.dataset_utils import prepare_numpy_dataset
from ai_models.hazard_traversability.model import SimpleHazardCNN, logits_to_outputs


LABEL_NAMES = {
    0: "safe",
    1: "moderate",
    2: "hazardous"
}


def run_inference_demo():
    X, y = prepare_numpy_dataset(split="train", num_samples=8, raw_limit=100)

    print("X shape:", X.shape)
    print("y shape:", y.shape)

    X_t = torch.tensor(X).permute(0, 3, 1, 2)

    model = SimpleHazardCNN(num_classes=3)
    model.eval()

    with torch.no_grad():
        logits = model(X_t)
        outputs = logits_to_outputs(logits)

    print("=== HAZARD MODEL INFERENCE DEMO ===")
    for i, out in enumerate(outputs):
        print(f"Sample {i}: true_label={LABEL_NAMES[int(y[i])]}, output={out}")

    # Evidence visualization
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for i in range(min(8, len(X))):
        axes[i].imshow(X[i])
        axes[i].axis("off")
        axes[i].set_title(
            f"True: {LABEL_NAMES[int(y[i])]}\n"
            f"Pred: {outputs[i]['hazard_type']}\n"
            f"Conf: {outputs[i]['confidence']:.2f}"
        )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_inference_demo()
