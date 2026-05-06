import argparse
import random
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import datasets, models, transforms


SCRIPT_DIR = Path(__file__).resolve().parent
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class Setup:
    name: str
    model_path: Path
    data_dir: Path


def get_default_setups():
    return [
        Setup(
            name="small",
            model_path=SCRIPT_DIR / "automerk_ai" / "carbrand_model.pth",
            data_dir=SCRIPT_DIR / "automerk_ai" / "dataset_small",
        ),
        Setup(
            name="medium",
            model_path=SCRIPT_DIR / "automerk_model.pth",
            data_dir=SCRIPT_DIR / "dataset_medium",
        ),
    ]


def parse_setups(values):
    if not values:
        return get_default_setups()

    setups = []
    for value in values:
        parts = value.split("|")
        if len(parts) != 3:
            raise ValueError(f"Ongeldige setup '{value}'. Gebruik formaat: naam|model_path|data_dir")
        setups.append(Setup(name=parts[0], model_path=Path(parts[1]), data_dir=Path(parts[2])))
    return setups


def resolve_test_dir(data_dir: Path) -> Path:
    test_dir = data_dir / "test"
    if not test_dir.exists():
        raise FileNotFoundError(f"Testmap niet gevonden: {test_dir}")
    return test_dir


def load_checkpoint(model_path: Path, device):
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        classes = checkpoint.get("classes")
        return state_dict, classes
    if isinstance(checkpoint, dict):
        return checkpoint, None
    raise ValueError(f"Onbekend checkpoint formaat: {model_path}")


def build_small_cnn_from_state_dict(state_dict):
    conv1_shape = state_dict["features.0.weight"].shape
    conv2_shape = state_dict["features.3.weight"].shape
    fc1_shape = state_dict["classifier.1.weight"].shape
    fc2_shape = state_dict["classifier.3.weight"].shape

    model = nn.Module()
    model.features = nn.Sequential(
        nn.Conv2d(conv1_shape[1], conv1_shape[0], kernel_size=conv1_shape[2]),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(conv2_shape[1], conv2_shape[0], kernel_size=conv2_shape[2]),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
    )
    model.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(fc1_shape[1], fc1_shape[0]),
        nn.ReLU(),
        nn.Linear(fc2_shape[1], fc2_shape[0]),
    )

    def forward(x):
        x = model.features(x)
        x = model.classifier(x)
        return x

    model.forward = forward
    return model


def build_model_for_state_dict(state_dict, class_count):
    keys = set(state_dict.keys())

    if "features.0.weight" in keys and "classifier.1.weight" in keys:
        return build_small_cnn_from_state_dict(state_dict)

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, class_count)
    return model


def sample_images(test_dir: Path, samples: int, seed: int):
    random.seed(seed)
    image_paths = []
    class_dirs = [d for d in test_dir.iterdir() if d.is_dir()]

    for class_dir in class_dirs:
        for image_path in class_dir.iterdir():
            if image_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                image_paths.append(image_path)

    if not image_paths:
        raise RuntimeError(f"Geen afbeeldingen gevonden in {test_dir}")

    samples = min(samples, len(image_paths))
    return random.sample(image_paths, samples)


def evaluate_setup(setup: Setup, sample_size: int, seed: int):
    if not setup.model_path.exists():
        return None, f"Model niet gevonden: {setup.model_path}"
    if not setup.data_dir.exists():
        return None, f"Dataset map niet gevonden: {setup.data_dir}"

    test_dir = resolve_test_dir(setup.data_dir)
    dataset = datasets.ImageFolder(test_dir)
    dataset_classes = dataset.classes

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    state_dict, checkpoint_classes = load_checkpoint(setup.model_path, device)
    class_names = checkpoint_classes if checkpoint_classes else dataset_classes

    model = build_model_for_state_dict(state_dict, len(class_names))
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    sampled_images = sample_images(test_dir, sample_size, seed)

    correct = 0
    total = 0
    details = []

    for image_path in sampled_images:
        true_label = image_path.parent.name
        image = Image.open(image_path).convert("RGB")
        tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1)
            conf, pred_idx = torch.max(probs, dim=1)

        pred_label = class_names[int(pred_idx.item())]
        is_correct = pred_label.lower() == true_label.lower()
        total += 1
        if is_correct:
            correct += 1

        details.append(
            f"{'✅' if is_correct else '❌'} {image_path.name} | echt={true_label} | voorspeld={pred_label} ({float(conf.item()) * 100:.1f}%)"
        )

    return {
        "name": setup.name,
        "correct": correct,
        "total": total,
        "details": details,
    }, None


def main():
    parser = argparse.ArgumentParser(description="Vergelijk modellen over datasetgroottes met x/10 score.")
    parser.add_argument(
        "--setup",
        action="append",
        help="Voeg setup toe: naam|model_path|data_dir (kan meerdere keren)",
    )
    parser.add_argument("--sample-size", type=int, default=10, help="Aantal testafbeeldingen per setup")
    parser.add_argument("--seed", type=int, default=42, help="Random seed voor reproduceerbare steekproef")
    parser.add_argument("--show-details", action="store_true", help="Toon voorspelling per afbeelding")
    args = parser.parse_args()

    setups = parse_setups(args.setup)

    print("Vergelijking datasetgroottes")
    print("-")

    for setup in setups:
        result, error = evaluate_setup(setup, args.sample_size, args.seed)
        if error:
            print(f"[{setup.name}] SKIP: {error}")
            continue

        print(f"[{result['name']}] score: {result['correct']}/{result['total']}")
        if args.show_details:
            for line in result["details"]:
                print("  " + line)
        print("-")


if __name__ == "__main__":
    main()
