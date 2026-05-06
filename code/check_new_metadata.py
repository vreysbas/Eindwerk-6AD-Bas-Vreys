from scipy.io import loadmat
import os

# Het pad dat je me gaf
TEST_MAT_PATH = r"C:\Users\vreys.bas\Desktop\automerk_ai\stanford_cars\cars_test_annos_withlabels.mat"
# We hebben ook de namen van de merken nodig uit het andere bestand
CLASS_NAMES_PATH = r"C:\Users\vreys.bas\Desktop\automerk_ai\stanford_cars\cars_annos.mat"

if not os.path.exists(TEST_MAT_PATH):
    print("Bestand nog niet gevonden. Check de spelling/locatie in PyCharm.")
else:
    # Laad de klassenamen (Audi, BMW, etc.)
    meta = loadmat(CLASS_NAMES_PATH)
    class_names = [str(x[0]) for x in meta["class_names"][0]]

    # Laad de test annotaties
    data = loadmat(TEST_MAT_PATH)
    annos = data['annotations'][0]

    print("--- Check van cars_test_annos_withlabels ---")
    for i in range(5):
        anno = annos[i]
        # In dit bestand heet het vaak 'fname' in plaats van 'relative_im_path'
        fname = anno['fname'][0] if 'fname' in anno.dtype.names else "onbekend"
        class_id = int(anno['class'][0][0]) - 1

        merk_naam = class_names[class_id]
        print(f"Bestand: {fname} is volgens dit bestand een: {merk_naam}")