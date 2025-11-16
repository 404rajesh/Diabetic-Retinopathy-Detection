# src/train.py
import os
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from dataset import DRDataset, get_transforms
from model import create_model
from utils import save_checkpoint, compute_qwk, plot_confusion
import utils
import random

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    running_loss = 0.0
    for imgs, labels in tqdm(loader, desc="Train", leave=False):
        imgs = imgs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
    return running_loss / len(loader.dataset)

def validate(model, loader, device):
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Val", leave=False):
            imgs = imgs.to(device)
            outputs = model(imgs)
            preds_batch = torch.argmax(outputs, dim=1).cpu().numpy()
            preds.extend(preds_batch.tolist())
            trues.extend(labels.numpy().tolist())
    return np.array(trues), np.array(preds)

def main(args):
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # read csv
    df = pd.read_csv(args.csv)
    # If notebook had paths like id_code only, keep as-is
    # create dataset
    transforms_train = get_transforms(train=True, size=args.img_size)
    transforms_val = get_transforms(train=False, size=args.img_size)

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    X = df['id_code'].values
    y = df['diagnosis'].values

    best_overall_qwk = -1.0

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n=== Fold {fold+1}/{args.folds} ===")
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)

        train_dataset = DRDataset(train_df, args.preproc_dir, transforms=transforms_train)
        val_dataset = DRDataset(val_df, args.preproc_dir, transforms=transforms_val)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

        model = create_model(model_name=args.model_name, num_classes=5, pretrained=True)
        model.to(device)

        # Stage 1: freeze backbone (if model supports)
        if args.stage1:
            for name, param in model.named_parameters():
                param.requires_grad = False
            # unfreeze classifier head
            for name, param in model.named_parameters():
                if 'fc' in name or 'classifier' in name or 'head' in name or name.endswith('weight') and 'fc' in name:
                    param.requires_grad = True

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-4)

        scaler = None

        best_qwk_fold = -1.0
        for epoch in range(1, args.epochs+1):
            print(f"Epoch {epoch}/{args.epochs}")
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler=scaler)
            print("Train loss:", train_loss)
            trues, preds = validate(model, val_loader, device)
            qwk = compute_qwk(trues, preds)
            print(f"Fold {fold} Epoch {epoch} QWK: {qwk:.4f}")

            # save best
            out_ckpt = os.path.join(args.ckpt_dir, f"best_fold{fold+1}.pth")
            if qwk > best_qwk_fold:
                best_qwk_fold = qwk
                save_checkpoint({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'qwk': qwk
                }, out_ckpt)
                print("Saved checkpoint:", out_ckpt)

            if qwk > best_overall_qwk:
                best_overall_qwk = qwk

            # optional LR step (simple)
            # you can add scheduler.step(qwk) etc.

        print(f"Best QWK for fold {fold+1}: {best_qwk_fold:.4f}")

    print("Training done. Best overall QWK:", best_overall_qwk)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="../data/train.csv", help="Path to train.csv")
    parser.add_argument("--preproc_dir", type=str, default="../preprocessed_data/train_images", help="Preprocessed image folder")
    parser.add_argument("--ckpt_dir", type=str, default="../checkpoints", help="Where to save checkpoints")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--model_name", type=str, default="resnet50")
    parser.add_argument("--stage1", action="store_true", help="Freeze backbone initially")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)
