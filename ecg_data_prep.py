import os
import wfdb
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ===== PATHS (absolute) =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(DATA_DIR, "ecg_images")     # where ECG images will be saved
CSV_OUT = os.path.join(DATA_DIR, "ecg_labels.csv")    # labels file

# ===== CONFIG =====
RECORDS = ["100", "101", "103"]    # sample MIT-BIH records
WINDOW_SIZE = 360                  # samples per beat window (~1 sec)
LEAD_INDEX = 0                     # use first ECG channel
FETCH_REMOTE = True                # download from PhysioNet automatically

# Normal beat annotations
NORMAL_SYMBOLS = ["N", "L", "R"]

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def main():
    # make sure folders exist
    ensure_dir(DATA_DIR)
    ensure_dir(OUTPUT_DIR)
    print("Saving images to:", OUTPUT_DIR, "| Exists:", os.path.isdir(OUTPUT_DIR))

    rows = []   # (filepath, label)

    for rec_name in RECORDS:
        print("Processing record:", rec_name)
        try:
            if FETCH_REMOTE:
                sig, fields = wfdb.rdsamp(rec_name, pn_dir="mitdb")
                ann = wfdb.rdann(rec_name, "atr", pn_dir="mitdb")
            else:
                sig, fields = wfdb.rdsamp(os.path.join("mitdb", rec_name))
                ann = wfdb.rdann(os.path.join("mitdb", rec_name), "atr")
        except Exception as e:
            print("Failed to load record", rec_name, ":", e)
            continue

        ecg = sig[:, LEAD_INDEX]   # one lead
        ann_samples = ann.sample
        ann_symbols = ann.symbol

        half = WINDOW_SIZE // 2

        for samp, sym in zip(ann_samples, ann_symbols):
            start = samp - half
            end = samp + half

            if start < 0 or end > len(ecg):
                continue

            label = "normal" if sym in NORMAL_SYMBOLS else "arrhythmia"

            window = ecg[start:end]
            window = window - np.mean(window)
            max_abs = np.max(np.abs(window))
            if max_abs != 0:
                window = window / max_abs

            fname = f"{rec_name}_{int(samp)}_{label}.png"
            fpath = os.path.join(OUTPUT_DIR, fname)

            # 🔴 SAFETY: ensure folder exists right before saving
            img_dir = os.path.dirname(fpath)
            ensure_dir(img_dir)
            # print("Saving:", fpath, "| Dir exists:", os.path.isdir(img_dir))

            plt.figure(figsize=(3, 3))
            plt.plot(window, linewidth=1)
            plt.axis("off")
            plt.margins(0)
            plt.savefig(fpath, bbox_inches="tight", pad_inches=0)
            plt.close()

            rows.append((fpath, label))

    df = pd.DataFrame(rows, columns=["filepath", "label"])
    df.to_csv(CSV_OUT, index=False)
    print("✔ Done — Saved", len(df), "ECG images and labels.")
    print("✔ Images folder:", OUTPUT_DIR)
    print("✔ Labels file:", CSV_OUT)

if __name__ == "__main__":
    main()
