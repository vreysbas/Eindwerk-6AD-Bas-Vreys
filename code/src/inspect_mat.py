from scipy.io import loadmat
import os

BASE = r"C:\Users\vreys.bas\Desktop\automerk_ai\dataset\StanfordCars-Dataset\stanford_cars"

TRAIN_ANNO = os.path.join(BASE, "cars_annos.mat")
TEST_ANNO  = os.path.join(BASE, "cars_test_annos_withlabels.mat")

def inspect(path):
    data = loadmat(path)
    print("\nFILE:", os.path.basename(path))
    print("Top-level keys:", [k for k in data.keys() if not k.startswith("__")])

    ann = data.get("annotations", None)
    if ann is None:
        print("❌ Geen 'annotations' key gevonden")
        return

    # ann is meestal array met structs
    first = ann[0][0]
    print("Type first annotation:", type(first))
    try:
        print("Dtype names (velden):", first.dtype.names)
    except Exception as e:
        print("Kan dtype names niet lezen:", e)

inspect(TRAIN_ANNO)
inspect(TEST_ANNO)
