from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox

import torch
import torch.nn as nn
from PIL import Image
from torchvision import datasets, models, transforms


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = SCRIPT_DIR / "automerk_model.pth"
DEFAULT_DATA_DIR = SCRIPT_DIR / "dataset_medium"


class InputPredictApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Automerk AI - Input veld")
        self.root.geometry("820x260")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        self.model = None
        self.class_names = []

        self.model_var = tk.StringVar(value=str(DEFAULT_MODEL_PATH))
        self.data_var = tk.StringVar(value=str(DEFAULT_DATA_DIR))
        self.image_var = tk.StringVar(value="")

        self.result_var = tk.StringVar(value="Nog geen voorspelling")
        self.conf_var = tk.StringVar(value="")

        self._build_ui()

    def _build_ui(self):
        container = tk.Frame(self.root, padx=12, pady=12)
        container.pack(fill=tk.BOTH, expand=True)

        tk.Label(container, text="Model (.pth):").grid(row=0, column=0, sticky="w")
        tk.Entry(container, textvariable=self.model_var, width=80).grid(row=0, column=1, sticky="ew", padx=6)
        tk.Button(container, text="Browse", command=self._pick_model).grid(row=0, column=2)

        tk.Label(container, text="Dataset map (voor klassen):").grid(row=1, column=0, sticky="w")
        tk.Entry(container, textvariable=self.data_var, width=80).grid(row=1, column=1, sticky="ew", padx=6)
        tk.Button(container, text="Browse", command=self._pick_dataset).grid(row=1, column=2)

        tk.Label(container, text="Afbeelding:").grid(row=2, column=0, sticky="w")
        tk.Entry(container, textvariable=self.image_var, width=80).grid(row=2, column=1, sticky="ew", padx=6)
        tk.Button(container, text="Browse", command=self._pick_image).grid(row=2, column=2)

        tk.Button(container, text="Model laden", command=self._load_model).grid(row=3, column=1, sticky="w", pady=(10, 4))
        tk.Button(container, text="Voorspel merk", command=self._predict).grid(row=3, column=1, sticky="e", pady=(10, 4))

        tk.Label(container, text="Voorspeld merk:", font=("Arial", 11, "bold")).grid(row=4, column=0, sticky="w", pady=(12, 0))
        tk.Label(container, textvariable=self.result_var, font=("Arial", 11, "bold")).grid(row=4, column=1, sticky="w", pady=(12, 0))

        tk.Label(container, textvariable=self.conf_var).grid(row=5, column=1, sticky="w", pady=(4, 0))

        container.columnconfigure(1, weight=1)

    def _pick_model(self):
        path = filedialog.askopenfilename(filetypes=[("PyTorch model", "*.pth"), ("All files", "*.*")])
        if path:
            self.model_var.set(path)

    def _pick_dataset(self):
        path = filedialog.askdirectory()
        if path:
            self.data_var.set(path)

    def _pick_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All files", "*.*")])
        if path:
            self.image_var.set(path)

    def _load_checkpoint(self, model_path: Path):
        checkpoint = torch.load(model_path, map_location=self.device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            return checkpoint["model_state_dict"], checkpoint.get("classes")
        if isinstance(checkpoint, dict):
            return checkpoint, None
        raise ValueError(f"Onbekend checkpoint formaat: {model_path}")

    def _resolve_class_folder(self, data_dir: Path) -> Path:
        test_dir = data_dir / "test"
        train_dir = data_dir / "train"
        if test_dir.exists():
            return test_dir
        if train_dir.exists():
            return train_dir
        raise FileNotFoundError("Geen 'test' of 'train' map gevonden in dataset map.")

    def _build_small_cnn_from_state_dict(self, state_dict):
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

    def _build_model_for_state_dict(self, state_dict):
        keys = set(state_dict.keys())
        if "features.0.weight" in keys and "classifier.1.weight" in keys:
            return self._build_small_cnn_from_state_dict(state_dict)

        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(self.class_names))
        return model

    def _load_model(self):
        try:
            model_path = Path(self.model_var.get())
            data_dir = Path(self.data_var.get())

            if not model_path.exists():
                raise FileNotFoundError(f"Model niet gevonden: {model_path}")
            if not data_dir.exists():
                raise FileNotFoundError(f"Dataset map niet gevonden: {data_dir}")

            class_folder = self._resolve_class_folder(data_dir)
            dataset = datasets.ImageFolder(class_folder)
            dataset_classes = dataset.classes

            state_dict, checkpoint_classes = self._load_checkpoint(model_path)
            self.class_names = checkpoint_classes if checkpoint_classes else dataset_classes

            model = self._build_model_for_state_dict(state_dict)
            model.load_state_dict(state_dict)
            model = model.to(self.device)
            model.eval()

            self.model = model
            messagebox.showinfo("OK", f"Model geladen met klassen: {self.class_names}")
        except Exception as exc:
            messagebox.showerror("Fout", str(exc))

    def _predict(self):
        try:
            if self.model is None:
                raise RuntimeError("Laad eerst een model.")

            image_path = Path(self.image_var.get())
            if not image_path.exists():
                raise FileNotFoundError("Kies eerst een geldige afbeelding.")

            image = Image.open(image_path).convert("RGB")
            tensor = self.transform(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                logits = self.model(tensor)
                probs = torch.softmax(logits, dim=1)
                conf, idx = torch.max(probs, dim=1)

            pred = self.class_names[int(idx.item())]
            confidence = float(conf.item()) * 100

            self.result_var.set(pred)
            self.conf_var.set(f"Confidence: {confidence:.2f}%")
        except Exception as exc:
            messagebox.showerror("Fout", str(exc))


def main():
    root = tk.Tk()
    InputPredictApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
