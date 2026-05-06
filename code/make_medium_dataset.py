import os
import shutil
import argparse
from pathlib import Path
from scipy.io import loadmat

# ==============================================================
# PADEN
# ==============================================================
# De basismap waar alles in staat
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_BASE_CANDIDATES = [
    SCRIPT_DIR / "stanford_cars",
    SCRIPT_DIR / "dataset" / "StanfordCars-Dataset" / "stanford_cars",
]


def detect_base_path() -> Path:
    for candidate in DEFAULT_BASE_CANDIDATES:
        if candidate.exists():
            return candidate
    return DEFAULT_BASE_CANDIDATES[0]


BASE_PATH = detect_base_path()

# 1. De lijst met merknamen (Audi, BMW, etc.) uit de devkit
META_MAT = os.path.join(BASE_PATH, "devkit", "cars_meta.mat")

# 2. De labels voor de training set (uit de devkit map)
TRAIN_MAT = os.path.join(BASE_PATH, "devkit", "cars_train_annos.mat")

# 3. De labels voor de test set (staat in de hoofdmap stanford_cars)
TEST_MAT = os.path.join(BASE_PATH, "cars_test_annos_withlabels.mat")

# Waar de nieuwe, schone dataset moet komen
OUTPUT_PATH = r"C:\Users\vreys.bas\Desktop\automerk_ai\dataset_medium"


def build_paths(base_path: Path, output_path: Path):
    base_path_str = str(base_path)
    output_path_str = str(output_path)

    meta_mat = os.path.join(base_path_str, "devkit", "cars_meta.mat")
    train_mat = os.path.join(base_path_str, "devkit", "cars_train_annos.mat")
    test_mat = os.path.join(base_path_str, "cars_test_annos_withlabels.mat")

    return base_path_str, output_path_str, meta_mat, train_mat, test_mat

# De merken die we willen filteren
BRANDS = ["Audi", "BMW", "Mercedes"]


def create_folders(output_path):
    """Maakt de mappen aan en wast ze eerst even wit (leegmaken)."""
    print("Mappen structuur voorbereiden...")
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for split in ["train", "test"]:
        for brand in BRANDS:
            path = os.path.join(output_path, split, brand)
            if os.path.exists(path):
                # Map leegmaken om oude 'foute' foto's te verwijderen
                shutil.rmtree(path)
            os.makedirs(path)


def process_set(mat_path, source_folder, split_name, class_names, base_path, output_path):
    """Koppelt de labels aan de juiste foto's en kopieert ze."""
    if not os.path.exists(mat_path):
        print(f"WAARSCHUWING: Kan {mat_path} niet vinden. Sla deze set over.")
        return 0

    data = loadmat(mat_path)
    annotations = data['annotations'][0]
    gekopieerd = 0

    print(f"Bezig met filteren van {split_name} set...")
    for anno in annotations:
        # In deze bestanden heet de naam 'fname'
        fname = str(anno['fname'][0])
        # De klasse index (bijv. 1 t/m 196) omzetten naar 0-based voor de lijst
        class_id = int(anno['class'][0][0]) - 1
        full_class_name = class_names[class_id]

        for brand in BRANDS:
            # Check of 'audi', 'bmw' of 'mercedes' in de naam van de klasse staat
            if brand.lower() in full_class_name.lower():
                src_path = os.path.join(base_path, source_folder, fname)
                dest_path = os.path.join(output_path, split_name, brand, fname)

                if os.path.exists(src_path):
                    shutil.copy(src_path, dest_path)
                    gekopieerd += 1
                break
    return gekopieerd


def main():
    parser = argparse.ArgumentParser(description="Bouw een 3-merken dataset (Audi/BMW/Mercedes) uit Stanford Cars.")
    parser.add_argument("--base-path", default=str(BASE_PATH), help="Pad naar de map met cars_train, cars_test en devkit")
    parser.add_argument("--output-path", default=str(SCRIPT_DIR / "dataset_medium"), help="Outputmap voor de gefilterde dataset")
    args = parser.parse_args()

    base_path = Path(args.base_path)
    output_path = Path(args.output_path)
    base_path_str, output_path_str, meta_mat, train_mat, test_mat = build_paths(base_path, output_path)

    print(f"Gebruik BASE_PATH: {base_path_str}")
    print(f"Gebruik OUTPUT_PATH: {output_path_str}")

    # Check of de belangrijkste bestanden er zijn
    if not os.path.exists(meta_mat):
        print(f"FOUT: {meta_mat} niet gevonden! Staat de 'devkit' map wel op de juiste plek?")
        return

    create_folders(output_path_str)

    # 1. Laad de echte namen van de 196 klassen
    meta = loadmat(meta_mat)
    class_names = [str(x[0]) for x in meta["class_names"][0]]

    # 2. Verwerk de Training foto's
    count_train = process_set(train_mat, "cars_train", "train", class_names, base_path_str, output_path_str)

    # 3. Verwerk de Test foto's
    count_test = process_set(test_mat, "cars_test", "test", class_names, base_path_str, output_path_str)

    print("-" * 40)
    print(f"Dataset succesvol opgebouwd!")
    print(f"- Audi/BMW/Mercedes in Train: {count_train}")
    print(f"- Audi/BMW/Mercedes in Test:  {count_test}")
    print(f"- Totaal aantal foto's:        {count_train + count_test}")
    print("-" * 40)
    print(f"Je kunt nu train.py gaan runnen met de map: {output_path_str}")


if __name__ == "__main__":
    main()