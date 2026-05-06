import os

# Pad naar jouw foto's
train_path = r"C:\Users\vreys.bas\Desktop\automerk_ai\stanford_cars\cars_train"

if os.path.exists(train_path):
    bestanden = os.listdir(train_path)[:10]
    print(f"Er staan {len(os.listdir(train_path))} bestanden in de map.")
    print("Dit zijn de eerste 10 namen:")
    for b in bestanden:
        print(f"- {b}")
else:
    print("Map niet gevonden!")