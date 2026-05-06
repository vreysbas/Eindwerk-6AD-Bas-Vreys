"""
Academische grafieken voor voertuigclassificatie (Audi, BMW, Mercedes).

Dit script genereert 4 afzonderlijke figuren met realistische dummy-data.
Vervang de dummy-data later door je eigen terminal-output uit train_small.py en evaluate.py.

Vereisten:
- matplotlib
- seaborn
- numpy

Installatie (indien nodig):
    pip install matplotlib seaborn numpy

Uitvoeren:
    python academische_grafieken.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# Algemene stijl voor een professionele, academische look.
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams["figure.dpi"] = 140


# Outputmap voor figuren.
OUTPUT_DIR = Path("figures_academisch")
OUTPUT_DIR.mkdir(exist_ok=True)

# Optioneel: zet hier een expliciet pad naar je eigen testfoto.
# Laat op None om automatisch een voorbeeldfoto uit de dataset te kiezen.
USER_IMAGE_PATH = None


def grafiek_1_leercurve() -> None:
    """
    Grafiek 1: Leercurve met dubbele y-as (twinx).

    TODO (jouw data plakken):
    - Vervang training_loss en val_accuracy_percent met waarden uit je terminal-output/logs.
    - Zorg dat beide lijsten even lang zijn en overeenkomen met hetzelfde epoch-nummer.
    """
    epochs = np.arange(1, 11)

    # Dummy-data gebaseerd op jouw beschrijving:
    # Training loss daalt van ongeveer 0.72 naar 0.08
    training_loss = [0.72, 0.61, 0.53, 0.44, 0.36, 0.29, 0.23, 0.17, 0.12, 0.08]

    # Validation accuracy stijgt van ongeveer 64% naar 89.9%
    val_accuracy_percent = [64.0, 69.5, 73.2, 76.8, 79.1, 81.5, 84.0, 86.2, 88.1, 89.9]

    fig, ax1 = plt.subplots(figsize=(9, 5.5))

    line1 = ax1.plot(
        epochs,
        training_loss,
        color="crimson",
        marker="o",
        linewidth=2.0,
        label="Training Loss",
    )
    ax1.set_xlabel("Epochs (1-10)")
    ax1.set_ylabel("Training Loss", color="crimson")
    ax1.tick_params(axis="y", labelcolor="crimson")

    ax2 = ax1.twinx()
    line2 = ax2.plot(
        epochs,
        val_accuracy_percent,
        color="forestgreen",
        marker="s",
        linewidth=2.0,
        label="Validation Accuracy (%)",
    )
    ax2.set_ylabel("Validation Accuracy (%)", color="forestgreen")
    ax2.tick_params(axis="y", labelcolor="forestgreen")

    ax1.set_title("Leercurve: Training Loss vs. Validation Accuracy")
    ax1.grid(True, linestyle="--", alpha=0.5)

    lines = line1 + line2
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="center right", frameon=True)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "grafiek_1_leercurve.png", bbox_inches="tight")
    plt.show()


def grafiek_2_confusion_matrix() -> None:
    """
    Grafiek 2: Confusion matrix heatmap.

    TODO (jouw data plakken):
    - Vervang cm met je eigen confusion matrix uit evaluate.py.
    - Volgorde klassen moet exact overeenkomen met labels.
    """
    labels = ["Audi", "BMW", "Mercedes"]

    # Dummy-confusion matrix met hoge accuraatheid,
    # plus gerichte verwarring: Audi -> BMW.
    cm = np.array(
        [
            [90, 8, 2],   # Werkelijk Audi
            [5, 92, 3],   # Werkelijk BMW
            [3, 4, 93],   # Werkelijk Mercedes
        ]
    )

    plt.figure(figsize=(7, 5.5))
    ax = sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=True,
        linewidths=0.5,
        linecolor="white",
    )

    ax.set_title("Confusion Matrix voor 3-klasse Voertuigclassificatie")
    ax.set_xlabel("Voorspelde Klasse (AI Output)")
    ax.set_ylabel("Werkelijke Klasse (Ground Truth)")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "grafiek_2_confusion_matrix.png", bbox_inches="tight")
    plt.show()


def grafiek_3_klassendistributie() -> None:
    """
    Grafiek 3: Klassendistributie in trainingsdata (bar chart).

    TODO (jouw data plakken):
    - Vervang class_counts met je echte aantallen per map/klasse.
    - De keys van class_counts zijn de klassennamen op de x-as.
    """
    class_counts = {
        "Audi": 468,
        "BMW": 492,
        "Mercedes": 479,
    }

    merken = list(class_counts.keys())
    aantallen = list(class_counts.values())

    plt.figure(figsize=(8, 5.5))
    bars = plt.bar(merken, aantallen, color="steelblue", edgecolor="black", linewidth=0.6)

    plt.title("Klassendistributie van de Trainingsset")
    plt.xlabel("Automerk")
    plt.ylabel("Aantal Trainingsafbeeldingen")
    plt.ylim(0, max(aantallen) + 60)

    for bar, value in zip(bars, aantallen):
        x = bar.get_x() + bar.get_width() / 2
        y = bar.get_height()
        plt.text(x, y + 5, f"{value}", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "grafiek_3_klassendistributie.png", bbox_inches="tight")
    plt.show()


def grafiek_4_transformatie_visualisatie() -> None:
    """
    Grafiek 4: Voor/Na data-transformatie (subplots).

    Deze versie gebruikt een echte autofoto uit je lokale dataset
    (Audi/BMW/Mercedes) in plaats van ruis.

    TODO (eigen lokale foto koppelen via torchvision.transforms):
    1) Installeer (indien nodig): pip install pillow torchvision
    2) Gebruik bijvoorbeeld:

       from PIL import Image
       from torchvision import transforms

       img_path = "pad/naar/jouw/testfoto.jpg"
       original = Image.open(img_path).convert("RGB")

       preprocess = transforms.Compose([
           transforms.Resize((224, 224)),
           transforms.ToTensor(),
           transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
       ])

       transformed_tensor = preprocess(original)  # shape: [3, 224, 224]

    3) Voor visualisatie van genormaliseerde tensor moet je meestal denormaliseren.
       In deze grafiek tonen we rechts een denormaliseerde preview na
       Resize(224x224) + Normalize, zodat het beeld menselijk leesbaar blijft.
    """
    def _find_sample_image() -> tuple[Path, str]:
        """Zoekt een representatieve autofoto in de lokale dataset."""
        if USER_IMAGE_PATH:
            custom_path = Path(USER_IMAGE_PATH)
            if custom_path.exists():
                return custom_path, "Custom"

        candidate_folders = [
            Path("dataset_medium/train/Audi"),
            Path("dataset_medium/train/BMW"),
            Path("dataset_medium/train/Mercedes"),
            Path("automerk_ai/dataset_medium/train/audi"),
            Path("automerk_ai/dataset_medium/train/bmw"),
            Path("automerk_ai/dataset_medium/train/mercedes"),
        ]
        extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")

        for folder in candidate_folders:
            if not folder.exists():
                continue
            for pattern in extensions:
                files = sorted(folder.glob(pattern))
                if files:
                    brand = folder.name.capitalize()
                    return files[0], brand

        raise FileNotFoundError(
            "Geen voorbeeldafbeelding gevonden. Zet USER_IMAGE_PATH op een geldig pad."
        )

    def _load_rgb_image(path: Path) -> np.ndarray:
        """Laadt beeld en normaliseert naar float32 [0, 1]."""
        img = plt.imread(path)
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0
        else:
            img = img.astype(np.float32)
            img = np.clip(img, 0.0, 1.0)

        # Grijsbeeld -> RGB
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)

        # RGBA -> RGB
        if img.shape[-1] == 4:
            img = img[..., :3]

        return img

    def _resize_nearest(image: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
        """Eenvoudige nearest-neighbor resize zonder extra dependencies."""
        h, w, _ = image.shape
        y_idx = np.linspace(0, h - 1, target_h).astype(int)
        x_idx = np.linspace(0, w - 1, target_w).astype(int)
        return image[y_idx][:, x_idx]

    image_path, brand_name = _find_sample_image()
    original_img = _load_rgb_image(image_path)

    resized_img = _resize_nearest(original_img, 224, 224)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    # Simuleert transforms.ToTensor() + transforms.Normalize(...)
    normalized_tensor = (resized_img - mean) / std

    # Voor visualisatie tonen we een denormaliseerde versie.
    transformed_img_for_plot = np.clip(normalized_tensor * std + mean, 0.0, 1.0)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))

    axes[0].imshow(original_img)
    axes[0].set_title(f"Originele Afbeelding ({brand_name})")
    axes[0].axis("off")

    axes[1].imshow(transformed_img_for_plot)
    axes[1].set_title("Na PyTorch Transformatie (224x224 + Normalisatie)")
    axes[1].axis("off")

    fig.suptitle(
        f"Visualisatie van Data-Transformatie - Voorbeeld: {brand_name}",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "grafiek_4_transformatie.png", bbox_inches="tight")
    plt.show()


def main() -> None:
    grafiek_1_leercurve()
    grafiek_2_confusion_matrix()
    grafiek_3_klassendistributie()
    grafiek_4_transformatie_visualisatie()


if __name__ == "__main__":
    main()