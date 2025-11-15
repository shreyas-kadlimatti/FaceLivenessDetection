# %% 1. IMPORT LIBRARIES

import os
import cv2
import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm
from skimage.feature import local_binary_pattern
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    roc_curve, classification_report
)
import matplotlib.pyplot as plt
import random

# Safe import to avoid "float object not callable" error
from sklearn.metrics import auc as sk_auc

print("✅ All libraries imported successfully.")

# %% 2. HAAR CASCADE FACE DETECTION & VISUALIZATION

# --- PATH SETUP ---
ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(ROOT, "data")
HAAR_PATH = os.path.join(ROOT, "haar", "haarcascade_frontalface_default.xml")

# --- INITIALIZE HAAR CASCADE ---
face_cascade = cv2.CascadeClassifier(HAAR_PATH)
if face_cascade.empty():
    raise FileNotFoundError("⚠️ Haar cascade file not found or empty. Make sure it's downloaded correctly.")

# --- FACE DETECTION FUNCTION ---
def detect_and_crop_face(img_path, cascade):
    """Detects the largest face in the image, crops, and returns grayscale + cropped face."""
    img = cv2.imread(img_path)
    if img is None:
        print(f"[Warning] Could not read image: {img_path}")
        return None, None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    if len(faces) == 0:
        # fallback center crop if no detection
        print(f"[Info] No face detected in {os.path.basename(img_path)} — using center crop.")
        h, w = gray.shape
        side = min(h, w)
        y0 = (h - side) // 2
        x0 = (w - side) // 2
        face = gray[y0:y0+side, x0:x0+side]
    else:
        # take the largest detected face
        x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
        face = gray[y:y+h, x:x+w]

    face = cv2.resize(face, (128, 128))
    return gray, face

# --- SAFE HELPERS TO AVOID .DS_Store ERRORS ---
def get_first_valid_subfolder(path):
    """Return first subfolder ignoring .DS_Store and non-directory files."""
    for item in os.listdir(path):
        full_path = os.path.join(path, item)
        if os.path.isdir(full_path):
            return full_path
    raise FileNotFoundError(f"No valid subfolder found in {path}")

def get_first_image(folder_path):
    """Return first image (jpg/jpeg/png) from a given folder."""
    for item in os.listdir(folder_path):
        if item.lower().endswith(('.jpg', '.jpeg', '.png')):
            return os.path.join(folder_path, item)
    raise FileNotFoundError(f"No image found in {folder_path}")

# --- DEFINE DATA PATHS ---
client_folder = os.path.join(DATA_DIR, "ClientRaw")
imposter_folder = os.path.join(DATA_DIR, "ImposterRaw")

client_subfolder = get_first_valid_subfolder(client_folder)
imposter_subfolder = get_first_valid_subfolder(imposter_folder)

sample_real = get_first_image(client_subfolder)
sample_fake = get_first_image(imposter_subfolder)

print("✅ Sample Real Image Path:", sample_real)
print("✅ Sample Fake Image Path:", sample_fake)

# --- DETECT & CROP FACES ---
gray_real, face_real = detect_and_crop_face(sample_real, face_cascade)
gray_fake, face_fake = detect_and_crop_face(sample_fake, face_cascade)

# --- VISUALIZATION ---
fig, axes = plt.subplots(2, 2, figsize=(8, 6))

axes[0, 0].imshow(cv2.cvtColor(cv2.imread(sample_real), cv2.COLOR_BGR2RGB))
axes[0, 0].set_title("Original Real")
axes[0, 1].imshow(face_real, cmap="gray")
axes[0, 1].set_title("Detected Face (Real)")

axes[1, 0].imshow(cv2.cvtColor(cv2.imread(sample_fake), cv2.COLOR_BGR2RGB))
axes[1, 0].set_title("Original Fake")
axes[1, 1].imshow(face_fake, cmap="gray")
axes[1, 1].set_title("Detected Face (Fake)")

for ax in axes.ravel():
    ax.axis("off")

plt.tight_layout()
plt.show()

# %% 3. Trial for LBP FEATURE EXTRACTION & VISUALIZATION

# Parameters for LBP
P = 8      # number of circularly symmetric neighbour set points
R = 1      # radius of circle
METHOD = 'uniform'

def extract_lbp_features(gray_face):
    """Return LBP image and normalized histogram features."""
    lbp = local_binary_pattern(gray_face, P, R, METHOD)
    # 59 bins for uniform pattern (P*(P-1)+3)
    n_bins = int(P * (P - 1) + 3)
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    return lbp, hist.astype(np.float32)

# Extract LBP for one real and one fake
lbp_real, hist_real = extract_lbp_features(face_real)
lbp_fake, hist_fake = extract_lbp_features(face_fake)

print(f"LBP histogram (real): {hist_real.shape}, sum = {hist_real.sum():.2f}")
print(f"LBP histogram (fake): {hist_fake.shape}, sum = {hist_fake.sum():.2f}")

# --- Visualization ---
fig, axes = plt.subplots(2, 2, figsize=(8, 6))

axes[0, 0].imshow(face_real, cmap='gray')
axes[0, 0].set_title("Original Grayscale (Real)")
axes[0, 1].imshow(lbp_real, cmap='gray')
axes[0, 1].set_title("LBP Image (Real)")

axes[1, 0].imshow(face_fake, cmap='gray')
axes[1, 0].set_title("Original Grayscale (Fake)")
axes[1, 1].imshow(lbp_fake, cmap='gray')
axes[1, 1].set_title("LBP Image (Fake)")

for ax in axes.ravel():
    ax.axis('off')

plt.tight_layout()
plt.show()

# %% 4. LBP + SVM
# -------------------------------------------------------------
#  Function to load images from all subfolders and extract LBP
# -------------------------------------------------------------
def load_images_and_extract_lbp(folder_path, label, max_images_per_folder=100):
    """
    Loads images from subfolders, applies Haar face detection,
    extracts LBP histogram features, and assigns labels.
    Args:
        folder_path : path to ClientRaw or ImposterRaw
        label        : 1 for real (client), 0 for fake (imposter)
        max_images_per_folder : limit for quick training
    """
    features, labels = [], []
    subfolders = [f for f in os.listdir(folder_path) if not f.startswith(".")]

    for sub in tqdm(subfolders, desc=f"Processing {os.path.basename(folder_path)}"):
        sub_path = os.path.join(folder_path, sub)
        if not os.path.isdir(sub_path):
            continue

        images = [img for img in os.listdir(sub_path)
                  if img.lower().endswith(('.jpg', '.jpeg', '.png'))]

        # ---------- QUICK MODE ----------
        if max_images_per_folder is not None and len(images) > max_images_per_folder:
            images = random.sample(images, max_images_per_folder)   # limit images for speed
        # ---------- FULL DATASET MODE ----------
        # To use the full dataset later, just comment the line above ↑

        for img_name in images:
            img_path = os.path.join(sub_path, img_name)
            _, face = detect_and_crop_face(img_path, face_cascade)
            if face is None:
                continue
            _, hist = extract_lbp_features(face)
            features.append(hist)
            labels.append(label)

    return np.array(features, dtype=np.float32), np.array(labels, dtype=np.int32)


# -------------------------------------------------------------
#  Load dataset and extract features
# -------------------------------------------------------------
client_folder = os.path.join(DATA_DIR, "ClientRaw")
imposter_folder = os.path.join(DATA_DIR, "ImposterRaw")

print("⚙️ Extracting LBP features for all real and fake faces...")

X_real, y_real = load_images_and_extract_lbp(client_folder, 1, max_images_per_folder=100)
X_fake, y_fake = load_images_and_extract_lbp(imposter_folder, 0, max_images_per_folder=100)

# ✅ Uncomment these two lines below later for FULL DATASET training
# X_real, y_real = load_images_and_extract_lbp(client_folder, 1, max_images_per_folder=None)
# X_fake, y_fake = load_images_and_extract_lbp(imposter_folder, 0, max_images_per_folder=None)

# -------------------------------------------------------------
#  Combine and Split
# -------------------------------------------------------------
X = np.vstack((X_real, X_fake))
y = np.hstack((y_real, y_fake))

print(f"✅ Feature extraction done. Total samples: {len(X)}, Feature length: {X.shape[1]}")

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# -------------------------------------------------------------
#  Train SVM Model (RBF Kernel)
# -------------------------------------------------------------
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

clf = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="rbf", probability=True))
])

print("🚀 Training SVM model on LBP features...")
clf.fit(X_train, y_train)

# -------------------------------------------------------------
#  Evaluate Model
# -------------------------------------------------------------
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report
)

y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]

acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec  = recall_score(y_test, y_pred)
f1   = f1_score(y_test, y_pred)
auc  = roc_auc_score(y_test, y_proba)

# -------------------------------------------------------------
#  📊 Display Evaluation Results as Table
# -------------------------------------------------------------
import pandas as pd

results_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC'],
    'Value': [acc, prec, rec, f1, auc]
})

print("\n📊 Evaluation Summary (LBP + SVM):")
display(results_df.style
        .set_caption("Model Performance Summary")
        .format({'Value': '{:.4f}'}))

# -------------------------------------------------------------
#  Save Trained Model
# -------------------------------------------------------------
MODEL_PATH = os.path.join(ROOT, "models", "svm_lbp.pkl")
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
import joblib
joblib.dump(clf, MODEL_PATH)
print(f"\n💾 Model saved to: {MODEL_PATH}")
 
# %% 5. PLOT ROC CURVE AND SAVE IMAGE

# Compute ROC curve points
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = sk_auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve — LBP + SVM')
plt.legend(loc="lower right")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

# Save ROC curve image
roc_dir = os.path.join(ROOT, "results", "roc_curves")
os.makedirs(roc_dir, exist_ok=True)
roc_path = os.path.join(roc_dir, "roc_curve_lbp_svm.png")
plt.savefig(roc_path, dpi=300)
plt.show()

print(f"✅ ROC curve saved to: {roc_path}")

# %% 6. DCT FEATURE EXTRACTION + SVM CLASSIFICATION (LIMITED FOR SPEED)
# -------------------------------------------------------------
#  Function: Extract DCT features
# -------------------------------------------------------------
def extract_dct_features(gray_face, block_size=8, num_coeffs=64):
    """
    Apply Discrete Cosine Transform (DCT) to a grayscale face image.
    Flatten top-left frequency coefficients as feature vector.
    """
    gray_float = np.float32(gray_face)
    dct = cv2.dct(gray_float)
    dct_low = dct[:block_size, :block_size]
    feat = dct_low.flatten()[:num_coeffs]
    feat = feat / np.linalg.norm(feat) if np.linalg.norm(feat) != 0 else feat
    return feat.astype(np.float32)

# -------------------------------------------------------------
#  Load dataset and extract DCT features
# -------------------------------------------------------------
def load_images_and_extract_dct(folder_path, label, max_images_per_folder=100):
    """
    Loads limited images from each subfolder and extracts DCT features.
    For full dataset training, comment out the limiting line below.
    """
    features, labels = [], []
    subfolders = [f for f in os.listdir(folder_path) if not f.startswith(".")]

    for sub in tqdm(subfolders, desc=f"Processing {os.path.basename(folder_path)} (DCT)"):
        sub_path = os.path.join(folder_path, sub)
        if not os.path.isdir(sub_path):
            continue

        images = [img for img in os.listdir(sub_path)
                  if img.lower().endswith(('.jpg', '.jpeg', '.png'))]

        # ✅ LIMIT 100 random IMAGES PER FOLDER (for speed)
        if max_images_per_folder is not None and len(images) > max_images_per_folder:
            images = random.sample(images, max_images_per_folder)
        # ❗ FOR FULL DATASET — comment the above line and uncomment below
        # images = images  # use all images

        for img_name in images:
            img_path = os.path.join(sub_path, img_name)
            _, face = detect_and_crop_face(img_path, face_cascade)
            if face is None:
                continue
            feat = extract_dct_features(face)
            features.append(feat)
            labels.append(label)

    return np.array(features, dtype=np.float32), np.array(labels, dtype=np.int32)


# -------------------------------------------------------------
#  Extract features for both classes
# -------------------------------------------------------------
print("\n⚙️ Extracting DCT features for real and fake faces...")

X_real_dct, y_real_dct = load_images_and_extract_dct(client_folder, 1, max_images_per_folder=100)
X_fake_dct, y_fake_dct = load_images_and_extract_dct(imposter_folder, 0, max_images_per_folder=100)

# ✅ FOR FULL DATASET (Uncomment below and comment the above two)
# X_real_dct, y_real_dct = load_images_and_extract_dct(client_folder, 1, max_images_per_folder=None)
# X_fake_dct, y_fake_dct = load_images_and_extract_dct(imposter_folder, 0, max_images_per_folder=None)

# -------------------------------------------------------------
#  Combine and Split
# -------------------------------------------------------------
X_dct = np.vstack((X_real_dct, X_fake_dct))
y_dct = np.hstack((y_real_dct, y_fake_dct))

print(f"✅ DCT feature extraction done. Total samples: {len(X_dct)}, Feature length: {X_dct.shape[1]}")

X_train_dct, X_test_dct, y_train_dct, y_test_dct = train_test_split(
    X_dct, y_dct, test_size=0.2, stratify=y_dct, random_state=42
)

# -------------------------------------------------------------
#  Train SVM (RBF Kernel)
# -------------------------------------------------------------
clf_dct = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="rbf", probability=True))
])

print("🚀 Training SVM on DCT features...")
clf_dct.fit(X_train_dct, y_train_dct)

# -------------------------------------------------------------
#  Evaluate Model
# -------------------------------------------------------------
y_pred_dct = clf_dct.predict(X_test_dct)
y_proba_dct = clf_dct.predict_proba(X_test_dct)[:, 1]

acc_dct  = accuracy_score(y_test_dct, y_pred_dct)
prec_dct = precision_score(y_test_dct, y_pred_dct)
rec_dct  = recall_score(y_test_dct, y_pred_dct)
f1_dct   = f1_score(y_test_dct, y_pred_dct)
auc_dct  = roc_auc_score(y_test_dct, y_proba_dct)

# -------------------------------------------------------------
#  📊 Display Evaluation Table
# -------------------------------------------------------------
results_dct = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC'],
    'Value': [acc_dct, prec_dct, rec_dct, f1_dct, auc_dct]
})

print("\n📊 Evaluation Summary (DCT + SVM):")
display(results_dct.style
        .set_caption("Model Performance Summary — DCT + SVM")
        .format({'Value': '{:.4f}'}))

# -------------------------------------------------------------
#  ROC Curve Plot and Save
# -------------------------------------------------------------
fpr_dct, tpr_dct, _ = roc_curve(y_test_dct, y_proba_dct)
roc_auc_dct = sk_auc(fpr_dct, tpr_dct)

plt.figure(figsize=(7, 6))
plt.plot(fpr_dct, tpr_dct, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc_dct:.4f})')
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve — DCT + SVM')
plt.legend(loc="lower right")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

roc_dir = os.path.join(ROOT, "results", "roc_curves")
os.makedirs(roc_dir, exist_ok=True)
roc_path_dct = os.path.join(roc_dir, "roc_curve_dct_svm.png")
plt.savefig(roc_path_dct, dpi=300)
plt.show()

print(f"✅ ROC curve saved to: {roc_path_dct}")

# -------------------------------------------------------------
#  Save Model
# -------------------------------------------------------------
MODEL_PATH_DCT = os.path.join(ROOT, "models", "svm_dct.pkl")
os.makedirs(os.path.dirname(MODEL_PATH_DCT), exist_ok=True)
joblib.dump(clf_dct, MODEL_PATH_DCT)
print(f"💾 Model saved to: {MODEL_PATH_DCT}")

# %% 7. LBP + DCT COMBINED FEATURES + SVM CLASSIFICATION
# -------------------------------------------------------------
#  Function to extract both LBP + DCT features from one face
# -------------------------------------------------------------
def extract_lbp_dct_features(gray_face):
    """Extract both LBP and DCT features from one face and concatenate."""
    # LBP features
    lbp, hist = extract_lbp_features(gray_face)
    # DCT features
    dct_feat = extract_dct_features(gray_face)
    # Combine both
    combined = np.hstack((hist, dct_feat))
    return combined.astype(np.float32)

# -------------------------------------------------------------
#  Load all faces and extract combined features
# -------------------------------------------------------------
def load_images_and_extract_lbp_dct(folder_path, label, max_images_per_folder=100):
    """
    Load limited images, extract both LBP and DCT features, combine them.
    """
    features, labels = [], []
    subfolders = [f for f in os.listdir(folder_path) if not f.startswith(".")]

    for sub in tqdm(subfolders, desc=f"Processing {os.path.basename(folder_path)} (LBP+DCT)"):
        sub_path = os.path.join(folder_path, sub)
        if not os.path.isdir(sub_path):
            continue

        images = [img for img in os.listdir(sub_path)
                  if img.lower().endswith(('.jpg', '.jpeg', '.png'))]

        # Limit per subfolder for speed
        if max_images_per_folder is not None and len(images) > max_images_per_folder:
            images = random.sample(images, max_images_per_folder)
        #  For full dataset: comment above & uncomment below
        # images = images

        for img_name in images:
            img_path = os.path.join(sub_path, img_name)
            _, face = detect_and_crop_face(img_path, face_cascade)
            if face is None:
                continue
            feat = extract_lbp_dct_features(face)
            features.append(feat)
            labels.append(label)

    return np.array(features, dtype=np.float32), np.array(labels, dtype=np.int32)


# -------------------------------------------------------------
#  Extract features for both real & fake faces
# -------------------------------------------------------------
print("\n⚙️ Extracting combined LBP + DCT features...")

X_real_comb, y_real_comb = load_images_and_extract_lbp_dct(client_folder, 1, max_images_per_folder=100)
X_fake_comb, y_fake_comb = load_images_and_extract_lbp_dct(imposter_folder, 0, max_images_per_folder=100)

# For full dataset (uncomment below, comment above two lines)
# X_real_comb, y_real_comb = load_images_and_extract_lbp_dct(client_folder, 1, max_images_per_folder=None)
# X_fake_comb, y_fake_comb = load_images_and_extract_lbp_dct(imposter_folder, 0, max_images_per_folder=None)

# Combine both
X_comb = np.vstack((X_real_comb, X_fake_comb))
y_comb = np.hstack((y_real_comb, y_fake_comb))

print(f"Combined features ready. Total samples: {len(X_comb)}, Feature length: {X_comb.shape[1]}")

# -------------------------------------------------------------
#  Train-Test Split
# -------------------------------------------------------------
X_train_comb, X_test_comb, y_train_comb, y_test_comb = train_test_split(
    X_comb, y_comb, test_size=0.2, stratify=y_comb, random_state=42
)

# -------------------------------------------------------------
#  Train SVM
# -------------------------------------------------------------
clf_comb = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="rbf", probability=True))
])

print("🚀 Training SVM on combined LBP + DCT features...")
clf_comb.fit(X_train_comb, y_train_comb)

# -------------------------------------------------------------
#  Evaluate
# -------------------------------------------------------------
y_pred_comb = clf_comb.predict(X_test_comb)
y_proba_comb = clf_comb.predict_proba(X_test_comb)[:, 1]

acc_comb  = accuracy_score(y_test_comb, y_pred_comb)
prec_comb = precision_score(y_test_comb, y_pred_comb)
rec_comb  = recall_score(y_test_comb, y_pred_comb)
f1_comb   = f1_score(y_test_comb, y_pred_comb)
auc_comb  = roc_auc_score(y_test_comb, y_proba_comb)

# -------------------------------------------------------------
#  📊 Display Evaluation Table
# -------------------------------------------------------------
results_comb = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC'],
    'Value': [acc_comb, prec_comb, rec_comb, f1_comb, auc_comb]
})

print("\n📊 Evaluation Summary (LBP + DCT + SVM):")
display(results_comb.style
        .set_caption("Model Performance Summary — LBP + DCT + SVM")
        .format({'Value': '{:.4f}'}))

# -------------------------------------------------------------
#  ROC Curve
# -------------------------------------------------------------
fpr_comb, tpr_comb, _ = roc_curve(y_test_comb, y_proba_comb)
roc_auc_comb = sk_auc(fpr_comb, tpr_comb)

plt.figure(figsize=(7, 6))
plt.plot(fpr_comb, tpr_comb, color='purple', lw=2, label=f'ROC Curve (AUC = {roc_auc_comb:.4f})')
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve — LBP + DCT + SVM')
plt.legend(loc="lower right")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

roc_dir = os.path.join(ROOT, "results", "roc_curves")
os.makedirs(roc_dir, exist_ok=True)
roc_path_comb = os.path.join(roc_dir, "roc_curve_lbp_dct_svm.png")
plt.savefig(roc_path_comb, dpi=300)
plt.show()

print(f"✅ ROC curve saved to: {roc_path_comb}")

# -------------------------------------------------------------
#  Save Model
# -------------------------------------------------------------
MODEL_PATH_COMB = os.path.join(ROOT, "models", "svm_lbp_dct.pkl")
os.makedirs(os.path.dirname(MODEL_PATH_COMB), exist_ok=True)
joblib.dump(clf_comb, MODEL_PATH_COMB)
print(f"💾 Model saved to: {MODEL_PATH_COMB}")

# %% 8. FINAL MODEL COMPARISON TABLE + GRAPH

# ---- COLLECT METRICS FROM ALL 3 MODELS ----
comparison_df = pd.DataFrame({
    "Model": ["LBP + SVM", "DCT + SVM", "Hybrid (LBP + DCT + SVM)"],
    "Accuracy": [acc, acc_dct, acc_comb],
    "Precision": [prec, prec_dct, prec_comb],
    "Recall": [rec, rec_dct, rec_comb],
    "F1 Score": [f1, f1_dct, f1_comb],
    "ROC AUC": [auc, auc_dct, auc_comb]
})

print("\n📌 FINAL COMPARISON TABLE")
display(
    comparison_df.style.format({
        "Accuracy": "{:.4f}",
        "Precision": "{:.4f}",
        "Recall": "{:.4f}",
        "F1 Score": "{:.4f}",
        "ROC AUC": "{:.4f}"
    })
)
# ---- BAR GRAPH FOR ACCURACY COMPARISON ----

plt.figure(figsize=(8,5))
bars = plt.bar(
    comparison_df["Model"],
    comparison_df["Accuracy"],
    color=["blue", "orange", "purple"]
)

# ---- Add value labels on bars ----
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2,
        height + 0.002,    # slight offset above bar
        f"{height:.4f}",   # formatting accuracy value
        ha='center',
        va='bottom',
        fontsize=10,
        fontweight='bold'
    )

plt.title("Accuracy Comparison of All Models")
plt.ylabel("Accuracy")
plt.ylim(0.85, 1.02)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
# %%
