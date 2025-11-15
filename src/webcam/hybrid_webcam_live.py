import cv2
import joblib
import numpy as np
from skimage.feature import local_binary_pattern

# --------------------------------------------------
# LOAD HYBRID MODEL (LBP + DCT + SVM)
# --------------------------------------------------
MODEL_PATH = "models/svm_lbp_dct.pkl"
clf = joblib.load(MODEL_PATH)
print("✅ Loaded Hybrid LBP + DCT + SVM model")

# --------------------------------------------------
# LOAD HAAR CASCADE FOR FACE DETECTION
# --------------------------------------------------
CASCADE_PATH = "haar/haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

# --------------------------------------------------
# LBP PARAMETERS + FUNCTION
# --------------------------------------------------
P = 8  # number of circularly symmetric neighbor set points
R = 1  # radius
METHOD = "uniform"

def extract_lbp_features(gray_face):
    lbp = local_binary_pattern(gray_face, P, R, METHOD)
    n_bins = int(P * (P - 1) + 3)
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    return hist.astype(np.float32)

# --------------------------------------------------
# DCT FEATURE EXTRACTOR
# --------------------------------------------------
def extract_dct_features(gray_face, block_size=8, num_coeffs=64):
    gray_float = np.float32(gray_face)
    dct = cv2.dct(gray_float)
    dct_low = dct[:block_size, :block_size]
    feat = dct_low.flatten()[:num_coeffs]
    # Normalize
    norm = np.linalg.norm(feat)
    if norm != 0:
        feat = feat / norm
    return feat.astype(np.float32)

# --------------------------------------------------
# HYBRID FEATURE EXTRACTOR (LBP + DCT)
# --------------------------------------------------
def extract_hybrid_features(gray_face):
    lbp_hist = extract_lbp_features(gray_face)
    dct_feat = extract_dct_features(gray_face)
    combined = np.hstack((lbp_hist, dct_feat))
    return combined.astype(np.float32)

# --------------------------------------------------
# START WEBCAM
# --------------------------------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Webcam could not be opened.")
    exit()

print("🎥 Hybrid Model Live Prediction Started... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to capture frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (128, 128))

        # Extract hybrid features
        features = extract_hybrid_features(face)

        # Predict
        proba = clf.predict_proba([features])[0]
        fake_prob = proba[0]  # class 0
        real_prob = proba[1]  # class 1

        label = "REAL" if real_prob > fake_prob else "FAKE"
        confidence = max(real_prob, fake_prob) * 100

        # Box color
        color = (0, 255, 0) if label == "REAL" else (0, 0, 255)

        # Draw rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        # Put text
        text = f"{label} ({confidence:.2f}%)"
        cv2.putText(frame, text, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("Hybrid LBP + DCT + SVM — Live Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()