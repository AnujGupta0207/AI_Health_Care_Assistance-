# ecg_predict.py
import os
import sys
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "ecg_model.h5")
IMG_SIZE = (128, 128)

# Load model (safe for inference)
if not os.path.exists(MODEL_PATH):
    print("❌ Model file not found at:", MODEL_PATH)
    sys.exit(1)

print("🔹 Loading model from:", MODEL_PATH)
model = tf.keras.models.load_model(MODEL_PATH, compile=False)


def predict_raw(image_path):
    """Return raw sigmoid value (float) from the model for the image."""
    img = load_img(image_path, color_mode="grayscale", target_size=IMG_SIZE)
    arr = img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    pred = model.predict(arr)[0][0]  # scalar
    return float(pred)


def infer_output_meaning(sample_dir):
    """
    Try to infer whether model output = P(normal) or P(arrhythmia).
    Strategy:
      - Find up to N images whose filename contains 'normal'
      - Find up to N images whose filename contains 'arrhythmia'
      - Compute avg predictions for both groups
      - If avg_normal > avg_arr -> model output is likely P(normal)
        else -> model output is likely P(arrhythmia)
    Returns:
      mapping: dict with keys:
         'pred_is' : 'P(normal)' or 'P(arrhythmia)'
         'avg_normal', 'avg_arr'
    """
    normal_paths = glob.glob(os.path.join(sample_dir, "*normal*.png"))
    arr_paths = glob.glob(os.path.join(sample_dir, "*arrhythmia*.png"))

    # if not found try jpg/jpeg
    if not normal_paths:
        normal_paths = glob.glob(os.path.join(sample_dir, "*normal*.jpg")) + glob.glob(os.path.join(sample_dir, "*normal*.jpeg"))
    if not arr_paths:
        arr_paths = glob.glob(os.path.join(sample_dir, "*arrhythmia*.jpg")) + glob.glob(os.path.join(sample_dir, "*arrhythmia*.jpeg"))

    N = 10
    normal_paths = normal_paths[:N]
    arr_paths = arr_paths[:N]

    # if we don't have both classes, return None
    if not normal_paths or not arr_paths:
        return None

    normal_preds = [predict_raw(p) for p in normal_paths]
    arr_preds = [predict_raw(p) for p in arr_paths]

    avg_normal = float(np.mean(normal_preds))
    avg_arr = float(np.mean(arr_preds))

    if avg_normal > avg_arr:
        pred_is = "P(normal)"
    else:
        pred_is = "P(arrhythmia)"

    return {"pred_is": pred_is, "avg_normal": avg_normal, "avg_arr": avg_arr}


def interpret_prediction(pred, mapping):
    """
    Given raw pred and mapping (from infer_output_meaning),
    return label and human-friendly score meaning.
    We will output:
      - score (0..1) always as "probability of ARRHYTHMIA" for clarity,
        and label accordingly.
    """
    if mapping is None:
        # Unknown mapping: assume model predicts P(normal) (common)
        # So pred -> P(normal); P(arr) = 1 - pred
        p_normal = pred
        p_arr = 1.0 - pred
    else:
        if mapping["pred_is"] == "P(normal)":
            p_normal = pred
            p_arr = 1.0 - pred
        else:  # model outputs P(arrhythmia)
            p_arr = pred
            p_normal = 1.0 - pred

    label = "ARRHYTHMIA" if p_arr >= 0.5 else "NORMAL"
    return label, p_arr, p_normal


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ecg_predict.py <path_to_ecg_image.png>")
        print("Optional (auto-detect): If you have a folder of images with 'normal' and 'arrhythmia' in filenames")
        print("  python ecg_predict.py <image.png>  # will try to auto-detect mapping from data/ecg_images/")
        sys.exit(0)

    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print("❌ Image not found:", image_path)
        sys.exit(1)

    # try to infer mapping from data/ecg_images
    sample_dir = os.path.join(BASE_DIR, "data", "ecg_images")
    mapping = None
    if os.path.isdir(sample_dir):
        mapping = infer_output_meaning(sample_dir)
        if mapping is None:
            print("⚠ Could not auto-detect mapping (need both 'normal' and 'arrhythmia' sample files).")
        else:
            print("🔎 Auto-detected model output meaning:", mapping["pred_is"])
            print(f"   avg_normal = {mapping['avg_normal']:.4f}, avg_arrhythmia = {mapping['avg_arr']:.4f}")

    raw_pred = predict_raw(image_path)
    label, p_arr, p_norm = interpret_prediction(raw_pred, mapping)

    print("--------------------------------------")
    print("Image:", image_path)
    print(f"Raw model output: {raw_pred:.4f}")
    print(f"Interpreted score: P(arrhythmia) = {p_arr:.4f}, P(normal) = {p_norm:.4f}")
    print("Final classification:", label)
    print("--------------------------------------")
    print("Note: This is a demo model. Always consult a cardiologist for real diagnosis.")

