from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "thesis_visuals"
OUT.mkdir(exist_ok=True)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    m = sum(values) / len(values)
    var = sum((x - m) ** 2 for x in values) / len(values)
    return m, math.sqrt(var)


def plot_medium_accuracy_distribution() -> None:
    path = ROOT / "accuracy_scatterplot_medium_points.csv"
    rows = read_csv_rows(path)

    grouped: dict[int, list[float]] = defaultdict(list)
    for r in rows:
        grouped[int(r["sample_size"])].append(float(r["accuracy_percent"]))

    sizes = sorted(grouped.keys())
    means = []
    stds = []
    for s in sizes:
        m, sd = mean_std(grouped[s])
        means.append(m)
        stds.append(sd)

    plt.figure(figsize=(10, 5.6))
    plt.errorbar(sizes, means, yerr=stds, marker="o", capsize=4, linewidth=2)
    plt.title("Medium setup: nauwkeurigheid per samplegrootte (gemiddelde +- std)")
    plt.xlabel("Samplegrootte")
    plt.ylabel("Nauwkeurigheid (%)")
    plt.ylim(0, 100)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUT / "figuur_gemiddelde_en_spreiding_medium.png", dpi=200)
    plt.close()



def plot_small_vs_medium_sample() -> None:
    path = ROOT / "accuracy_scatterplot_sample_zoom_test.csv"
    rows = read_csv_rows(path)

    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    setup_to_sizes: dict[str, set[int]] = defaultdict(set)
    setups = set()
    for r in rows:
        setup = r["setup"]
        x = int(float(r["x_value"]))
        y = float(r["accuracy_percent"])
        grouped[(setup, x)].append(y)
        setup_to_sizes[setup].add(x)
        setups.add(setup)

    plt.figure(figsize=(10, 5.6))
    for setup in sorted(setups):
        xs = sorted(setup_to_sizes[setup])
        ys = []
        for x in xs:
            vals = grouped[(setup, x)]
            ys.append(sum(vals) / len(vals))
        plt.plot(xs, ys, marker="o", linewidth=2, label=setup)

    plt.title("Vergelijking small vs medium per samplegrootte")
    plt.xlabel("Samplegrootte")
    plt.ylabel("Gemiddelde nauwkeurigheid (%)")
    plt.ylim(0, 100)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "figuur_small_vs_medium.png", dpi=200)
    plt.close()



def plot_training_size_curve() -> None:
    path = ROOT / "accuracy_scatterplot_train_fast.csv"
    rows = read_csv_rows(path)

    points = sorted(
        [(int(float(r["x_value"])), float(r["accuracy_percent"])) for r in rows],
        key=lambda p: p[0],
    )
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    plt.figure(figsize=(9.5, 5.4))
    plt.plot(xs, ys, marker="o", linewidth=2.2, color="#0b7285")
    for x, y in zip(xs, ys):
        plt.text(x, y + 1.0, f"{y:.1f}%", ha="center", fontsize=9)
    plt.title("Effect van trainingsgrootte op nauwkeurigheid")
    plt.xlabel("Aantal trainingsafbeeldingen (x_value)")
    plt.ylabel("Nauwkeurigheid (%)")
    plt.ylim(0, 100)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUT / "figuur_training_size_effect.png", dpi=200)
    plt.close()



def plot_7_brand_curve() -> None:
    path = ROOT / "accuracy_scatterplot_links_focus_7merken.csv"
    rows = read_csv_rows(path)

    points = sorted(
        [(int(float(r["x_value"])), float(r["accuracy_percent"])) for r in rows],
        key=lambda p: p[0],
    )
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    plt.figure(figsize=(9.5, 5.4))
    plt.plot(xs, ys, marker="o", linewidth=2.2, color="#5f3dc4")
    plt.fill_between(xs, ys, alpha=0.15, color="#5f3dc4")
    plt.title("7-merken scenario: nauwkeurigheid per samplegrootte")
    plt.xlabel("Samplegrootte (x_value)")
    plt.ylabel("Nauwkeurigheid (%)")
    plt.ylim(0, 100)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUT / "figuur_7merken_curve.png", dpi=200)
    plt.close()



def plot_class_distribution() -> None:
    train_root = ROOT / "dataset_medium" / "train"
    test_root = ROOT / "dataset_medium" / "test"

    classes = sorted([p.name for p in train_root.iterdir() if p.is_dir()])
    train_counts = []
    test_counts = []

    for c in classes:
        train_dir = train_root / c
        test_dir = test_root / c
        train_counts.append(len([p for p in train_dir.iterdir() if p.is_file()]))
        test_counts.append(len([p for p in test_dir.iterdir() if p.is_file()]))

    x = np.arange(len(classes))
    width = 0.38

    plt.figure(figsize=(11, 5.8))
    plt.bar(x - width / 2, train_counts, width=width, label="train")
    plt.bar(x + width / 2, test_counts, width=width, label="test")
    plt.xticks(x, classes, rotation=20)
    plt.title("Klassedistributie dataset_medium (train vs test)")
    plt.xlabel("Klasse")
    plt.ylabel("Aantal afbeeldingen")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "figuur_klassedistributie_train_test.png", dpi=200)
    plt.close()



def main() -> None:
    plot_medium_accuracy_distribution()
    plot_small_vs_medium_sample()
    plot_training_size_curve()
    plot_7_brand_curve()
    plot_class_distribution()
    print(f"Grafieken opgeslagen in: {OUT}")


if __name__ == "__main__":
    main()
