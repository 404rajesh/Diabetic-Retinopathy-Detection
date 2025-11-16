# src/preprocess.py
import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

def crop_image_from_gray(img, tol=7):
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol
        if mask.sum() == 0:
            return img
        return img[np.ix_(mask.any(1), mask.any(0))]

def enhance_image(img, size=224):
    # img expected RGB
    img = crop_image_from_gray(img)
    try:
        img = cv2.resize(img, (size, size))
    except Exception:
        # fallback to center crop + resize
        h, w = img.shape[:2]
        min_dim = min(h, w)
        startx = w//2 - min_dim//2
        starty = h//2 - min_dim//2
        img = img[starty:starty+min_dim, startx:startx+min_dim]
        img = cv2.resize(img, (size, size))
    # enhance contrast
    img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0,0), 10), -4, 128)
    return img

def preprocess_dataset(data_dir, out_dir, csv_path, img_col='id_code', ext='.png', size=224):
    """
    data_dir: folder with original images (train_images)
    out_dir: folder where processed images will be saved
    csv_path: path to train.csv describing ids
    """
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    ids = df[img_col].values
    pbar = tqdm(ids, desc="Preprocessing images")
    skipped = 0
    for id_code in pbar:
        src = os.path.join(data_dir, f"{id_code}{ext}")
        # try jpg or jpeg if png not present
        if not os.path.exists(src):
            for e in ['.png', '.jpg', '.jpeg']:
                tmp = os.path.join(data_dir, f"{id_code}{e}")
                if os.path.exists(tmp):
                    src = tmp
                    break
        if not os.path.exists(src):
            skipped += 1
            continue
        img = cv2.imread(src)
        if img is None:
            skipped += 1
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        proc = enhance_image(img, size=size)
        # write as PNG in RGB order -> cv2 expects BGR so convert back
        proc_bgr = cv2.cvtColor(proc, cv2.COLOR_RGB2BGR)
        out_path = os.path.join(out_dir, f"{id_code}.png")
        cv2.imwrite(out_path, proc_bgr)
    print(f"Preprocessing done. Skipped: {skipped}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Original train_images folder")
    parser.add_argument("--out_dir", type=str, required=True, help="Output preprocessed folder")
    parser.add_argument("--csv", type=str, required=True, help="Path to train.csv")
    parser.add_argument("--size", type=int, default=224)
    args = parser.parse_args()
    preprocess_dataset(args.data_dir, args.out_dir, args.csv, size=args.size)
