from pathlib import Path
import shutil

import scipy.io


ROOT = Path(__file__).resolve().parent
STANFORD_ROOT = ROOT / "stanford_cars"
META_PATH = STANFORD_ROOT / "devkit" / "cars_meta.mat"
TRAIN_ANNOS_PATH = STANFORD_ROOT / "devkit" / "cars_train_annos.mat"
TEST_ANNOS_PATH = STANFORD_ROOT / "cars_test_annos_withlabels.mat"
TRAIN_SRC = STANFORD_ROOT / "cars_train"
TEST_SRC = STANFORD_ROOT / "cars_test"

TRAIN_PATH = ROOT / "dataset_medium" / "train"
TEST_PATH = ROOT / "dataset_medium" / "test"

TARGET_BRANDS = ["jeep", "toyota", "nissan", "volkswagen"]


def normalize(text: str) -> str:
    return " ".join(text.lower().replace("-", " ").split())


def brand_from_classname(classname: str):
    low = normalize(classname)
    for brand in TARGET_BRANDS:
        if brand in low:
            return brand
    return None


def main():
    if not META_PATH.exists():
        raise FileNotFoundError(f"Niet gevonden: {META_PATH}")
    if not TRAIN_ANNOS_PATH.exists():
        raise FileNotFoundError(f"Niet gevonden: {TRAIN_ANNOS_PATH}")
    if not TEST_ANNOS_PATH.exists():
        raise FileNotFoundError(f"Niet gevonden: {TEST_ANNOS_PATH}")
    if not TRAIN_SRC.exists() or not TEST_SRC.exists():
        raise FileNotFoundError("cars_train/cars_test mappen niet gevonden in stanford_cars")

    train_data = scipy.io.loadmat(str(TRAIN_ANNOS_PATH))
    train_annotations = train_data["annotations"][0]
    test_data = scipy.io.loadmat(str(TEST_ANNOS_PATH))
    test_annotations = test_data["annotations"][0]

    meta = scipy.io.loadmat(str(META_PATH))
    class_names = [str(x[0]) for x in meta["class_names"][0]]

    for brand in TARGET_BRANDS:
        (TRAIN_PATH / brand).mkdir(parents=True, exist_ok=True)
        (TEST_PATH / brand).mkdir(parents=True, exist_ok=True)

    counters = {
        "train": {brand: 0 for brand in TARGET_BRANDS},
        "test": {brand: 0 for brand in TARGET_BRANDS},
    }

    for anno in train_annotations:
        fname = str(anno["fname"][0])
        class_id = int(anno["class"][0][0])

        class_name = class_names[class_id - 1]
        brand = brand_from_classname(class_name)
        if brand is None:
            continue

        src = TRAIN_SRC / fname
        if not src.exists():
            continue

        dst_dir = TRAIN_PATH / brand
        shutil.copy2(src, dst_dir / src.name)
        counters["train"][brand] += 1

    for anno in test_annotations:
        fname = str(anno["fname"][0])
        class_id = int(anno["class"][0][0])

        class_name = class_names[class_id - 1]
        brand = brand_from_classname(class_name)
        if brand is None:
            continue

        src = TEST_SRC / fname
        if not src.exists():
            continue

        dst_dir = TEST_PATH / brand
        shutil.copy2(src, dst_dir / src.name)
        counters["test"][brand] += 1

    print("Klaar. Gekopieerde aantallen:")
    print("train:", counters["train"])
    print("test :", counters["test"])


if __name__ == "__main__":
    main()