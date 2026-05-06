import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

from compare_dataset_sizes import (
    SUPPORTED_EXTENSIONS,
    build_model_for_state_dict,
    load_checkpoint,
    parse_setups,
    resolve_test_dir,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SAMPLE_SIZES = [10, 20, 30, 50, 75, 100, 150, 200, 300, 500, 800, 1000]


def parse_sample_sizes(raw: str):
    values = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        size = int(token)
        if size <= 0:
            raise ValueError(f"Sample size moet > 0 zijn, gekregen: {size}")
        values.append(size)
    if not values:
        raise ValueError("Geen geldige sample sizes opgegeven")
    return sorted(set(values))


def collect_correct_flags(setup):
    if not setup.model_path.exists():
        raise FileNotFoundError(f"Model niet gevonden: {setup.model_path}")
    if not setup.data_dir.exists():
        raise FileNotFoundError(f"Dataset map niet gevonden: {setup.data_dir}")

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

    image_paths = []
    class_dirs = sorted([d for d in test_dir.iterdir() if d.is_dir()], key=lambda p: p.name.lower())
    for class_dir in class_dirs:
        for image_path in sorted(class_dir.iterdir(), key=lambda p: p.name.lower()):
            if image_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                image_paths.append(image_path)

    if not image_paths:
        raise RuntimeError(f"Geen afbeeldingen gevonden in {test_dir}")

    flags = []
    with torch.no_grad():
        for image_path in image_paths:
            true_label = image_path.parent.name
            image = Image.open(image_path).convert("RGB")
            tensor = transform(image).unsqueeze(0).to(device)

            logits = model(tensor)
            pred_idx = int(logits.argmax(dim=1).item())
            pred_label = class_names[pred_idx]
            flags.append(pred_label.lower() == true_label.lower())

    return flags, device


def sample_indices_balanced(targets, class_count, sample_size, rng):
    by_class = {class_idx: [] for class_idx in range(class_count)}
    for index, class_idx in enumerate(targets):
        by_class[class_idx].append(index)

    per_class = sample_size // class_count
    remainder = sample_size % class_count
    chosen = []

    for class_idx in range(class_count):
        k = min(per_class, len(by_class[class_idx]))
        if k > 0:
            chosen.extend(rng.sample(by_class[class_idx], k))

    class_order = list(range(class_count))
    rng.shuffle(class_order)
    remaining_pool = {c: [idx for idx in by_class[c] if idx not in set(chosen)] for c in by_class}

    for class_idx in class_order:
        if remainder <= 0:
            break
        if remaining_pool[class_idx]:
            chosen.append(rng.choice(remaining_pool[class_idx]))
            remainder -= 1

    if len(chosen) < sample_size:
        all_indices = list(range(len(targets)))
        need = sample_size - len(chosen)
        chosen_set = set(chosen)
        candidates = [idx for idx in all_indices if idx not in chosen_set]
        if candidates:
            chosen.extend(rng.sample(candidates, min(need, len(candidates))))

    return chosen


def evaluate_subset_training(setup, training_sizes, repeats, seed, epochs, batch_size, learning_rate, test_samples=0):
    if not setup.data_dir.exists():
        raise FileNotFoundError(f"Dataset map niet gevonden: {setup.data_dir}")

    train_dir = setup.data_dir / "train"
    test_dir = setup.data_dir / "test"
    if not train_dir.exists() or not test_dir.exists():
        raise FileNotFoundError(f"Train/test mappen niet gevonden in: {setup.data_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=transform)
    class_count = len(train_dataset.classes)
    targets = list(train_dataset.targets)
    rng = random.Random(seed)

    if test_samples and test_samples > 0:
        sample_count = min(test_samples, len(test_dataset))
        sampled_test_indices = rng.sample(list(range(len(test_dataset))), sample_count)
        eval_test_dataset = Subset(test_dataset, sampled_test_indices)
    else:
        eval_test_dataset = test_dataset

    test_loader = DataLoader(eval_test_dataset, batch_size=batch_size, shuffle=False)
    points = []

    for training_size in training_sizes:
        actual_size = min(training_size, len(train_dataset))
        if actual_size <= 0:
            continue

        for _ in range(repeats):
            subset_indices = sample_indices_balanced(targets, class_count, actual_size, rng)
            train_subset = Subset(train_dataset, subset_indices)
            train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)

            model = models.resnet18(weights="IMAGENET1K_V1")
            model.fc = nn.Linear(model.fc.in_features, class_count)
            model = model.to(device)

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)

            model.train()
            for _epoch in range(epochs):
                for images, labels in train_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in test_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = model(images)
                    preds = outputs.argmax(dim=1)
                    total += labels.size(0)
                    correct += (preds == labels).sum().item()

            accuracy = 100.0 * correct / total if total else 0.0
            points.append((actual_size, accuracy))

    return points, device, len(train_dataset), len(eval_test_dataset)


def build_scatter_points(correct_flags, sample_sizes, repeats, seed):
    rng = random.Random(seed)
    total_available = len(correct_flags)
    points = []

    for sample_size in sample_sizes:
        actual_size = min(sample_size, total_available)
        if actual_size <= 0:
            continue

        for _ in range(repeats):
            chosen = rng.sample(correct_flags, actual_size)
            accuracy = 100.0 * sum(chosen) / actual_size
            points.append((actual_size, accuracy))

    return points


def save_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("setup,x_value,accuracy_percent\n")
        for setup_name, x_value, accuracy in rows:
            handle.write(f"{setup_name},{x_value},{accuracy:.4f}\n")


def plot_full(setup_points, mode, output_path):
    fig, ax = plt.subplots(1, 1, figsize=(10, 6.5))

    for setup_name, points, extra_label in setup_points:
        xs = [x for x, _ in points]
        ys = [y for _, y in points]
        ax.scatter(xs, ys, s=20, alpha=0.45, label=f"{setup_name} {extra_label}")

        grouped = {}
        for x, y in points:
            grouped.setdefault(x, []).append(y)
        mean_x = sorted(grouped)
        mean_y = [sum(grouped[x]) / len(grouped[x]) for x in mean_x]
        ax.plot(mean_x, mean_y, linewidth=1.8)

    x_label = "Aantal afbeeldingen in training" if mode == "train" else "Aantal afbeeldingen in steekproef"
    ax.set_title("Volledig bereik")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Nauwkeurigheid (%)")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.25)
    ax.legend()

    title_mode = "traininggrootte" if mode == "train" else "steekproefgrootte"
    fig.suptitle(f"Nauwkeurigheid vs {title_mode} (scatterplot)")
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Maak een scatterplot van nauwkeurigheid (%) over het volledige bereik.")
    parser.add_argument(
        "--setup",
        action="append",
        help="Voeg setup toe: naam|model_path|data_dir (kan meerdere keren)",
    )
    parser.add_argument(
        "--mode",
        choices=["sample", "train"],
        default="train",
        help="sample: x-as = steekproefgrootte op testset, train: x-as = aantal trainingsafbeeldingen",
    )
    parser.add_argument(
        "--sample-sizes",
        default=",".join(str(v) for v in DEFAULT_SAMPLE_SIZES),
        help="Komma-gescheiden x-waardes, bv: 10,20,30,50,100,200",
    )
    parser.add_argument("--repeats", type=int, default=20, help="Aantal metingen per sample size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--epochs", type=int, default=2, help="Alleen in train mode: epochs per training")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate in train mode")
    parser.add_argument(
        "--test-samples",
        type=int,
        default=0,
        help="Alleen in train mode: aantal testafbeeldingen per meting (0 = volledige testset)",
    )
    parser.add_argument("--zoom-max", type=int, default=120, help="Niet gebruikt (oude optie, behouden voor compatibiliteit)")
    parser.add_argument(
        "--output",
        default=str(SCRIPT_DIR / "accuracy_scatterplot_training.png"),
        help="Pad naar output scatterplot PNG",
    )
    parser.add_argument(
        "--csv-output",
        default=str(SCRIPT_DIR / "accuracy_scatterplot_points.csv"),
        help="Pad naar CSV met alle punten",
    )
    args = parser.parse_args()

    if args.repeats <= 0:
        raise ValueError("--repeats moet > 0 zijn")
    if args.test_samples < 0:
        raise ValueError("--test-samples moet >= 0 zijn")

    sample_sizes = parse_sample_sizes(args.sample_sizes)
    setups = parse_setups(args.setup)

    all_rows = []
    setup_points = []

    for setup in setups:
        if args.mode == "train":
            points, device, train_count, test_count = evaluate_subset_training(
                setup=setup,
                training_sizes=sample_sizes,
                repeats=args.repeats,
                seed=args.seed,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.lr,
                test_samples=args.test_samples,
            )
            extra_label = f"(train={train_count}, test={test_count})"
            print(f"[{setup.name}] Mode=train, device={device}, train={train_count}, test={test_count}, punten={len(points)}")
        else:
            correct_flags, device = collect_correct_flags(setup)
            points = build_scatter_points(correct_flags, sample_sizes, args.repeats, args.seed)
            extra_label = f"(test={len(correct_flags)})"
            print(f"[{setup.name}] Mode=sample, device={device}, test={len(correct_flags)}, punten={len(points)}")

        all_rows.extend((setup.name, x_value, accuracy) for x_value, accuracy in points)
        setup_points.append((setup.name, points, extra_label))

    output_path = Path(args.output)
    plot_full(setup_points, args.mode, output_path)

    csv_path = Path(args.csv_output)
    save_csv(csv_path, all_rows)

    print(f"Scatterplot opgeslagen: {output_path}")
    print(f"Punten CSV opgeslagen: {csv_path}")


if __name__ == "__main__":
    main()