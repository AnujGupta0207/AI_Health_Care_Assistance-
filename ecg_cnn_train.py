import os
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# === Paths ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "data", "ecg_labels.csv")

# === Load labels ===
df = pd.read_csv(CSV_PATH)

# Make filepaths absolute
def make_abs(path):
    if os.path.isabs(path):
        return path
    return os.path.join(BASE_DIR, path)

df["filepath"] = df["filepath"].apply(make_abs)

print("Total ECG images:", len(df))
print(df["label"].value_counts())

# === Train/validation split ===
train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

print("Train size:", len(train_df), "Val size:", len(val_df))

# === Image generators ===
img_size = (128, 128)
batch_size = 32

train_datagen = ImageDataGenerator(rescale=1.0/255.0)
val_datagen = ImageDataGenerator(rescale=1.0/255.0)

train_gen = train_datagen.flow_from_dataframe(
    train_df,
    x_col="filepath",
    y_col="label",
    target_size=img_size,
    color_mode="grayscale",
    class_mode="binary",
    batch_size=batch_size,
    shuffle=True
)

val_gen = val_datagen.flow_from_dataframe(
    val_df,
    x_col="filepath",
    y_col="label",
    target_size=img_size,
    color_mode="grayscale",
    class_mode="binary",
    batch_size=batch_size,
    shuffle=False
)

print("Class indices from generator:", train_gen.class_indices)  # e.g. {'arrhythmia': 0, 'normal': 1}

# === Handle class imbalance with class weights ===
n_arr = (train_df["label"] == "arrhythmia").sum()
n_norm = (train_df["label"] == "normal").sum()
total = n_arr + n_norm

w_arr = total / (2 * n_arr)
w_norm = total / (2 * n_norm)

class_weight = {
    train_gen.class_indices["arrhythmia"]: w_arr,
    train_gen.class_indices["normal"]: w_norm,
}

print("Using class weights:", class_weight)
print("Counts -> arrhythmia:", n_arr, ", normal:", n_norm)

# === CNN model ===
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(img_size[0], img_size[1], 1)),
    tf.keras.layers.Conv2D(16, (3, 3), activation="relu"),
    tf.keras.layers.MaxPool2D(2, 2),

    tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
    tf.keras.layers.MaxPool2D(2, 2),

    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.MaxPool2D(2, 2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")  # binary: normal vs arrhythmia
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# === Train ===
EPOCHS = 8  # can adjust later

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    class_weight=class_weight  # <--- IMPORTANT
)

# === Save model ===
OUT_MODEL = os.path.join(BASE_DIR, "ecg_model.h5")
model.save(OUT_MODEL)
print("✅ Training complete. Model saved as", OUT_MODEL)
