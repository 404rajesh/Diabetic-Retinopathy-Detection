# src/evaluate.py
import os
import argparse
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import DRDataset, get_transforms
from model import create_model
from utils import compute_qwk, plot_confusion
from sklearn.metrics import classification_report, confusion_matrix

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Load CSV
    df = pd.read_csv(args.csv)

    # Dataset + loader
    transforms = get_transforms(train=False, size=args.img_size)
    dataset = DRDataset(df, args.preproc_dir, transforms)
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=2)

    # Load model
    model = create_model(args.model_name, num_classes=5, pretrained=False)
    ckpt = torch.load(args.weights, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Compute QWK
    qwk = compute_qwk(all_labels, all_preds)
    print("\n========== RESULTS ==========")
    print("QWK:", qwk)

    # Classification Report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, digits=4))

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("\nConfusion Matrix:")
    print(cm)

    # Save Confusion Matrix Image
    os.makedirs(args.output_dir, exist_ok=True)
    cm_path = os.path.join(args.output_dir, "confusion_matrix.png")
    plot_confusion(all_labels, all_preds, out_path=cm_path)
    print(f"Saved confusion matrix → {cm_path}")

    # Save predictions CSV
    out_csv = os.path.join(args.output_dir, "predictions.csv")
    df["predicted"] = all_preds
    df.to_csv(out_csv, index=False)
    print(f"Saved predictions → {out_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="data/train.csv")
    parser.add_argument("--preproc_dir", type=str, default="preprocessed_data/train_images")
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="efficientnet_b3")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--output_dir", type=str, default="eval_outputs")
    args = parser.parse_args()

    evaluate(args)
