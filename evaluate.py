import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SCRIPT_DIR / "dataset_medium"
DEFAULT_MODEL_PATH = SCRIPT_DIR / "automerk_model.pth"


def evaluate(model_path: str, data_dir: str, batch_size: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    test_dataset = datasets.ImageFolder(Path(data_dir) / "test", transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    class_count = len(test_dataset.classes)

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, class_count)

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()

    correct = 0
    total = 0
    per_class_correct = [0 for _ in range(class_count)]
    per_class_total = [0 for _ in range(class_count)]
    confusion = [[0 for _ in range(class_count)] for _ in range(class_count)]

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            preds = outputs.argmax(dim=1)

            total += labels.size(0)
            correct += (preds == labels).sum().item()

            for yi, pi in zip(labels.tolist(), preds.tolist()):
                per_class_total[yi] += 1
                if yi == pi:
                    per_class_correct[yi] += 1
                confusion[yi][pi] += 1

    overall_acc = 100 * correct / total if total else 0.0

    print(f"Device: {device}")
    print(f"Classes: {test_dataset.classes}")
    print(f"Samples (test): {total}")
    print(f"Overall accuracy: {overall_acc:.2f}%")

    print("\nPer-class accuracy:")
    for i, name in enumerate(test_dataset.classes):
        acc = (100 * per_class_correct[i] / per_class_total[i]) if per_class_total[i] else 0.0
        print(f"- {name}: {per_class_correct[i]}/{per_class_total[i]} ({acc:.2f}%)")

    print("\nConfusion matrix (rows=true, cols=pred):")
    header = " " * 14 + " ".join([f"{c:>10}" for c in test_dataset.classes])
    print(header)
    for i, row in enumerate(confusion):
        print(f"{test_dataset.classes[i]:>14} " + " ".join([f"{v:>10}" for v in row]))


def main():
    parser = argparse.ArgumentParser(description="Evalueer een getraind automerk-model op de testset.")
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH), help="Pad naar .pth modelbestand")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="Pad naar dataset met test map")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch grootte voor evaluatie")
    args = parser.parse_args()

    evaluate(args.model_path, args.data_dir, args.batch_size)


if __name__ == "__main__":
    main()
