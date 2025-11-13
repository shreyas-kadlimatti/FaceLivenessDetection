Face Liveness Detection Using LBP, DCT & Hybrid Features (SVM Classifier)

This project implements a Face Liveness Detection System using classical computer vision techniques — Local Binary Patterns (LBP), Discrete Cosine Transform (DCT), and a Hybrid LBP+DCT method combined with Support Vector Machine (SVM) classification.

It detects whether a face is real or fake (spoof) based on texture and frequency features extracted from face images.

🔧 Requirements

Install necessary packages:
pip install numpy pandas scikit-learn opencv-python matplotlib tqdm scikit-image joblib

🧠 Features Implemented

✔️ 1. LBP Feature Extraction + SVM

Texture-based pattern recognition.

✔️ 2. DCT Feature Extraction + SVM

Frequency-domain spoof detection.

✔️ 3. Hybrid LBP + DCT Features + SVM

Combination of both for maximum accuracy.

✔️ 4. Haar Cascade Face Detection

Face detection and cropping before feature extraction.

✔️ 5. Model Training Evaluation
	•	Accuracy
	•	Precision
	•	Recall
	•	F1 Score
	•	ROC-AUC
	•	ROC Curve plots saved inside results/roc_curves/

✔️ 6. Model Comparison Table + Bar Graph

Final accuracy comparison of all three trained models.

📦 Project Folder Structure

FaceLivenessDetection/
│
├── data/
│   ├── ClientRaw/      # Real images
│   ├── ImposterRaw/    # Fake images
│
├── haar/
│   └── haarcascade_frontalface_default.xml
│
├── models/
│   ├── svm_lbp.pkl
│   ├── svm_dct.pkl
│   ├── svm_lbp_dct.pkl
│
├── results/
│   └── roc_curves/     # ROC curve images
│
├── src/
│   ├── fld_experiments.py    # Main training & evaluation script
│   ├── utils_features.py
│   ├── utils_plot.py
│
└── README.md

Inside the project folder:
cd FaceLivenessDetection
python3 src/fld_experiments.py

👨‍💻 Author 
Shreyas Kadlimatti