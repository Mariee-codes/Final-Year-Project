"""Train the Crop Doctor disease classifier and report lab vs field accuracy.

Run this on Google Colab or Kaggle with a GPU (free tiers are enough).

1. Get PlantVillage (colour images), e.g. the Kaggle "PlantVillage dataset",
   and keep only the crops in scope: folders starting with Tomato___,
   Potato___ and Corn_(maize)___. Layout:  data/plantvillage/<class_name>/*.jpg
2. Put field photos (PlantDoc + your own Kenyan farm photos) in the SAME class
   folder names:  data/field/<class_name>/*.jpg   - never used for training
   unless you pass --finetune-field.
3. python ml/train_crop_doctor.py --data data/plantvillage --field data/field

Outputs (in models/):
  crop_doctor.onnx   model used by the API
  labels.json        class names in model order
  report.json        accuracy, precision, recall, F1 for lab AND field test sets
  confusion_lab.csv / confusion_field.csv
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
OUT = Path(__file__).resolve().parents[1] / "models"

# Strong augmentation: PlantVillage has plain lab backgrounds, so we push the
# model to look at the leaf, not the background (see Noyan 2022).
train_tf = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.05),
    transforms.RandomGrayscale(0.05),
    transforms.GaussianBlur(5, sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
    transforms.RandomErasing(p=0.3),
])
eval_tf = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(), transforms.Normalize(MEAN, STD),
])


def split_indices(n: int, val=0.15, test=0.15):
    idx = np.random.permutation(n)
    n_test, n_val = int(n * test), int(n * val)
    return idx[n_test + n_val:], idx[n_test:n_test + n_val], idx[:n_test]


def build_model(num_classes: int, arch: str) -> nn.Module:
    if arch == "mobilenet_v3":
        m = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
    else:
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
    return m


def run_epoch(model, loader, device, opt=None, crit=nn.CrossEntropyLoss(label_smoothing=0.1)):
    train = opt is not None
    model.train(train)
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.set_grad_enabled(train):
            out = model(x)
            loss = crit(out, y)
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
        loss_sum += loss.item() * len(y)
        correct += (out.argmax(1) == y).sum().item()
        total += len(y)
    return loss_sum / total, correct / total


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        ps.append(model(x.to(device)).argmax(1).cpu())
        ys.append(y)
    return torch.cat(ys).numpy(), torch.cat(ps).numpy()


def report(name, y, p, labels):
    present = sorted(set(y) | set(p))
    rep = classification_report(y, p, labels=present, target_names=[labels[i] for i in present],
                                output_dict=True, zero_division=0)
    cm = confusion_matrix(y, p, labels=list(range(len(labels))))
    np.savetxt(OUT / f"confusion_{name}.csv", cm, fmt="%d", delimiter=",",
               header=",".join(labels), comments="")
    print(f"\n=== {name.upper()} TEST SET ({len(y)} images) ===")
    print(f"accuracy: {rep['accuracy']:.4f}  macro F1: {rep['macro avg']['f1-score']:.4f}")
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="PlantVillage folder (class subfolders)")
    ap.add_argument("--field", help="Field photo folder with the same class names")
    ap.add_argument("--arch", default="efficientnet_b0", choices=["efficientnet_b0", "mobilenet_v3"])
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    full_train = datasets.ImageFolder(args.data, transform=train_tf)
    full_eval = datasets.ImageFolder(args.data, transform=eval_tf)
    labels = full_train.classes
    tr, va, te = split_indices(len(full_train))
    dl = lambda ds, idx, shuffle: DataLoader(Subset(ds, idx), batch_size=args.batch,
                                             shuffle=shuffle, num_workers=2)
    train_dl, val_dl, test_dl = dl(full_train, tr, True), dl(full_eval, va, False), dl(full_eval, te, False)

    model = build_model(len(labels), args.arch).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    best, best_state = 0.0, None
    for epoch in range(args.epochs):
        tl, ta = run_epoch(model, train_dl, device, opt)
        vl, vacc = run_epoch(model, val_dl, device)
        sched.step()
        print(f"epoch {epoch + 1}/{args.epochs}  train acc {ta:.3f}  val acc {vacc:.3f}")
        if vacc > best:
            best, best_state = vacc, {k: v.cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)

    results = {"arch": args.arch, "classes": labels,
               "lab": report("lab", *predict(model, test_dl, device), labels)}

    if args.field:
        field = datasets.ImageFolder(args.field, transform=eval_tf)
        # Map field class names onto the training label order.
        remap = {i: labels.index(c) for i, c in enumerate(field.classes) if c in labels}
        keep = [i for i, (_, y) in enumerate(field.samples) if y in remap]
        field.targets = [remap.get(t, -1) for t in field.targets]
        field.samples = [(p, remap.get(y, -1)) for p, y in field.samples]
        fdl = DataLoader(Subset(field, keep), batch_size=args.batch)
        results["field"] = report("field", *predict(model, fdl, device), labels)
        drop = results["lab"]["accuracy"] - results["field"]["accuracy"]
        print(f"\nLab-to-field accuracy drop: {drop * 100:.1f} percentage points (report this honestly)")

    with open(OUT / "report.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(OUT / "labels.json", "w") as f:
        json.dump(labels, f, indent=2)

    model.eval().cpu()
    torch.onnx.export(model, torch.randn(1, 3, 224, 224), OUT / "crop_doctor.onnx",
                      input_names=["image"], output_names=["logits"],
                      dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17)
    print(f"\nSaved model and reports to {OUT}")


if __name__ == "__main__":
    main()
