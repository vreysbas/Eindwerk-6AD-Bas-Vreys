from scipy.io import loadmat

MAT_PATH = r"C:\Users\vreys.bas\Desktop\automerk_ai\stanford_cars\cars_annos.mat"
data = loadmat(MAT_PATH)

# We kijken naar de structuur van één 'annotation'
sample_anno = data["annotations"][0][0]

print("--- Metadata Check ---")
for name in sample_anno.dtype.names:
    print(f"{name}: {sample_anno[name]}")