import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from rembg import remove
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.ensemble import (
    ExtraTreesRegressor,
    VotingRegressor
)
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
import joblib


# =========================
# PATH CONFIGURATION
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

IMAGE_FOLDER = os.path.join(PROJECT_DIR, "data", "images")
CSV_PATH = os.path.join(PROJECT_DIR, "data", "data.csv")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(IMAGE_FOLDER, exist_ok=True)


# =========================
# LOAD LABELS
# =========================

print("Loading data...")
df = pd.read_csv(CSV_PATH)
df.columns = df.columns.str.strip()

# Standardize column names
if 'image_filename' not in df.columns and 'filename' in df.columns:
    df['image_filename'] = df['filename']
if 'image_filename' not in df.columns:
    df.rename(columns={df.columns[0]: 'image_filename'}, inplace=True)

print(f"Samples: {len(df)}")
print(f"Columns: {df.columns.tolist()}")


# =========================
# BACKGROUND REMOVAL
# =========================

def remove_bg(path):
    """Remove background from leaf image"""
    try:
        with open(path, "rb") as f:
            data = f.read()
        out = remove(data)
        img = cv2.imdecode(np.frombuffer(out, np.uint8), cv2.IMREAD_UNCHANGED)
        if img.shape[2] == 4:
            alpha = img[:, :, 3]
            img = img[:, :, :3]
            img[alpha == 0] = 0
        return img
    except:
        return cv2.imread(path)


# =========================
# DATA AUGMENTATION
# =========================

def augment_image(img, aug_type=0):
    """Apply data augmentation transformations"""
    if aug_type == 0:
        return img
    elif aug_type == 1:
        return cv2.flip(img, 1)
    elif aug_type == 2:
        return cv2.flip(img, 0)
    elif aug_type == 3:
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        angle = np.random.uniform(-5, 5)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    return img


# =========================
# FEATURE EXTRACTION
# =========================

def extract_features(img):
    """Extract 22 features from leaf image"""
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    
    lower = np.array([30, 60, 60])
    upper = np.array([80, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    pixels = rgb[mask > 0]
    
    if len(pixels) == 0:
        return None
    
    # Color features (6)
    R = np.mean(pixels[:, 0])
    G = np.mean(pixels[:, 1])
    B = np.mean(pixels[:, 2])
    Rstd = np.std(pixels[:, 0])
    Gstd = np.std(pixels[:, 1])
    Bstd = np.std(pixels[:, 2])
    
    # Vegetation indices (6)
    ExG = 2*G - R - B
    VARI = (G - R) / (G + R - B + 1e-6)
    GLI = (2*G - R - B) / (2*G + R + B + 1e-6)
    NGI = G / (R + G + B + 1e-6)
    CIVE = 0.441*R - 0.881*G + 0.385*B
    
    h = np.mean(hsv[:, :, 0][mask > 0]) / 180
    s = np.mean(hsv[:, :, 1][mask > 0]) / 255
    v = np.mean(hsv[:, :, 2][mask > 0]) / 255
    DGCI = ((h - 0.33) + (1 - s) + (1 - v)) / 3
    
    # Texture features (5)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = (gray / 32).astype(np.uint8)
    
    glcm = graycomatrix(gray, distances=[1], angles=[0], levels=8, symmetric=True, normed=True)
    contrast = graycoprops(glcm, "contrast")[0, 0]
    homogeneity = graycoprops(glcm, "homogeneity")[0, 0]
    energy = graycoprops(glcm, "energy")[0, 0]
    dissimilarity = graycoprops(glcm, "dissimilarity")[0, 0]
    correlation = graycoprops(glcm, "correlation")[0, 0]
    
    # Shape & edge features (3)
    lbp = local_binary_pattern(gray, 8, 1, method='uniform')
    lbp_mean = np.mean(lbp)
    lbp_std = np.std(lbp)
    
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    
    # Morphological features (2)
    leaf_pixels = np.sum(mask > 0)
    leaf_area = leaf_pixels / (mask.size + 1e-6)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        cnt = max(contours, key=cv2.contourArea)
        moments = cv2.moments(cnt)
        if moments['m00'] != 0:
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            compactness = 4*np.pi*area / (perimeter**2 + 1e-6)
        else:
            compactness = 0
    else:
        compactness = 0
    
    return [R, G, B, Rstd, Gstd, Bstd, ExG, VARI, GLI, NGI, CIVE, DGCI,
            contrast, homogeneity, energy, dissimilarity, correlation,
            lbp_mean, lbp_std, edge_density, leaf_area, compactness]


# =========================
# BUILD DATASET WITH AUGMENTATION
# =========================

X = []
y = []

print("\nExtracting features...")

for i, row in tqdm(df.iterrows(), total=len(df)):
    filename = str(row["image_filename"]).strip()
    spad = row["spad_value"]
    
    if "(" in filename and " (" not in filename:
        filename = filename.replace("(", " (")
    
    path = os.path.join(IMAGE_FOLDER, filename)
    if not os.path.exists(path):
        continue
    
    try:
        img = cv2.imread(path)
        if img is None:
            continue
        
        img = remove_bg(path)
        
        # Original + 3 augmentations
        for aug_type in range(4):
            img_aug = augment_image(img, aug_type)
            f = extract_features(img_aug)
            if f is not None:
                X.append(f)
                y.append(spad)
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        continue

X = np.array(X)
y = np.array(y)

print(f"Usable samples: {len(X)}")


# =========================
# NORMALIZE TARGET VARIABLE
# =========================

from sklearn.preprocessing import MinMaxScaler

y_scaler = MinMaxScaler()
y_normalized = y_scaler.fit_transform(y.reshape(-1, 1)).flatten()

y_original = y.copy()
y = y_normalized

print("Target normalized to range [0, 1]")


# =========================
# ENSEMBLE MODEL
# =========================

voting_model = Pipeline([
    ('scale', StandardScaler()),
    ('voting', VotingRegressor(
        estimators=[
            ('et', ExtraTreesRegressor(
                n_estimators=800,
                max_depth=20,
                min_samples_split=3,
                min_samples_leaf=1,
                max_features='sqrt',
                random_state=42
            )),
            ('xgb', XGBRegressor(
                objective='reg:squarederror',
                n_estimators=400,
                learning_rate=0.02,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )),
            ('ridge', Ridge(alpha=1.0))
        ]
    ))
])


# =========================
# 5-FOLD CROSS VALIDATION
# =========================

kf = KFold(n_splits=5, shuffle=True, random_state=42)

all_r2 = []
all_mae = []
all_predictions = []
all_actuals = []
fold = 1

print("\n===== RESULTS (ENSEMBLE MODEL WITH AUGMENTATION) =====")

for train_idx, test_idx in kf.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    voting_model.fit(X_train, y_train)
    pred = voting_model.predict(X_test)
    
    pred_denorm = y_scaler.inverse_transform(pred.reshape(-1, 1)).flatten()
    y_test_denorm = y_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    
    r2 = r2_score(y_test_denorm, pred_denorm)
    mae = mean_absolute_error(y_test_denorm, pred_denorm)
    
    print(f"Fold {fold}: R2={r2:.3f} MAE={mae:.2f}")
    
    all_r2.append(r2)
    all_mae.append(mae)
    all_predictions.extend(pred_denorm)
    all_actuals.extend(y_test_denorm)
    fold += 1

all_predictions = np.array(all_predictions)
all_actuals = np.array(all_actuals)

print("\n**FINAL RESULTS**")
avg_r2 = round(np.mean(all_r2), 3)
avg_mae = round(np.mean(all_mae), 2)
print(f"Average R2: {avg_r2}")
print(f"Average MAE: {avg_mae}")
print(f"R2 Std Dev: {np.std(all_r2):.3f}")
print(f"MAE Std Dev: {np.std(all_mae):.2f}")


# =========================
# VISUALIZATION
# =========================

print("\nGenerating visualizations...")

residuals = all_actuals - all_predictions

# Plot 1: Actual vs Predicted
fig1, ax1 = plt.subplots(figsize=(10, 7))
ax1.scatter(all_actuals, all_predictions, alpha=0.6, color='blue', s=50)
min_val = min(all_actuals.min(), all_predictions.min())
max_val = max(all_actuals.max(), all_predictions.max())
ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
ax1.set_xlabel('Actual SPAD', fontsize=11)
ax1.set_ylabel('Predicted SPAD', fontsize=11)
ax1.set_title(f'Actual vs Predicted (R2={avg_r2}, MAE={avg_mae})', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '01_Actual_vs_Predicted.png'), dpi=150, bbox_inches='tight')
print("✓ Graph saved: 01_Actual_vs_Predicted.png")
plt.close(fig1)

# Plot 2: Residuals vs Predicted
fig2, ax2 = plt.subplots(figsize=(10, 7))
ax2.scatter(all_predictions, residuals, alpha=0.6, color='green', s=50)
ax2.axhline(y=0, color='r', linestyle='--', lw=2)
ax2.set_xlabel('Predicted SPAD', fontsize=11)
ax2.set_ylabel('Residuals', fontsize=11)
ax2.set_title('Residual Plot', fontsize=12, fontweight='bold')
ax2.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '02_Residual_Plot.png'), dpi=150, bbox_inches='tight')
print("✓ Graph saved: 02_Residual_Plot.png")
plt.close(fig2)

# Plot 3: Error Distribution
fig3, ax3 = plt.subplots(figsize=(10, 7))
errors = np.abs(residuals)
ax3.hist(errors, bins=20, color='orange', alpha=0.7, edgecolor='black')
ax3.axvline(avg_mae, color='r', linestyle='--', lw=2, label=f'Mean MAE={avg_mae}')
ax3.set_xlabel('Absolute Error', fontsize=11)
ax3.set_ylabel('Frequency', fontsize=11)
ax3.set_title('Error Distribution', fontsize=12, fontweight='bold')
ax3.legend()
ax3.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '03_Error_Distribution.png'), dpi=150, bbox_inches='tight')
print("✓ Graph saved: 03_Error_Distribution.png")
plt.close(fig3)

# Plot 4: Per-Fold Performance
fig4, ax4 = plt.subplots(figsize=(10, 7))
folds = np.arange(1, len(all_r2) + 1)
ax4_twin = ax4.twinx()
bars1 = ax4.bar(folds - 0.2, all_r2, width=0.4, label='R2', color='skyblue', alpha=0.8)
bars2 = ax4_twin.bar(folds + 0.2, all_mae, width=0.4, label='MAE', color='salmon', alpha=0.8)
ax4.set_xlabel('Fold', fontsize=11)
ax4.set_ylabel('R2 Score', fontsize=11, color='skyblue')
ax4_twin.set_ylabel('MAE', fontsize=11, color='salmon')
ax4.set_title('Per-Fold Performance', fontsize=12, fontweight='bold')
ax4.set_xticks(folds)
ax4.tick_params(axis='y', labelcolor='skyblue')
ax4_twin.tick_params(axis='y', labelcolor='salmon')
ax4.grid(alpha=0.3, axis='y')

lines1, labels1 = ax4.get_legend_handles_labels()
lines2, labels2 = ax4_twin.get_legend_handles_labels()
ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '04_Per_Fold_Performance.png'), dpi=150, bbox_inches='tight')
print("✓ Graph saved: 04_Per_Fold_Performance.png")
plt.close(fig4)


# =========================
# SAVE RESULTS
# =========================

results_df = pd.DataFrame({
    'Actual_SPAD': all_actuals,
    'Predicted_SPAD': all_predictions,
    'Absolute_Error': np.abs(residuals),
    'Relative_Error_Percent': 100 * np.abs(residuals) / (all_actuals + 1e-6)
})

results_df.to_csv(os.path.join(OUTPUT_DIR, 'predictions.csv'), index=False)
print("\n✓ Predictions saved: predictions.csv")

joblib.dump(voting_model, os.path.join(OUTPUT_DIR, 'model.pkl'))
print("✓ Model saved: model.pkl")

# Create report
report = f"""
=============================================
SPAD PREDICTION MODEL - FINAL REPORT
=============================================

MODEL SPECIFICATIONS:
  Features: 22 (Color, Vegetation, Texture, Morphological)
  Data Augmentation: 4x per sample
  Total Training Samples: {len(X)} ({len(X)//4} original x 4)
  Models Ensemble: 3 (ExtraTrees + XGBoost + Ridge)

PERFORMANCE METRICS (5-Fold Cross-Validation):
  Average R² Score: {avg_r2} (± {np.std(all_r2):.3f})
  Average MAE: {avg_mae} (± {np.std(all_mae):.2f})
  Min R² Score: {min(all_r2):.3f} | Max R² Score: {max(all_r2):.3f}
  Min MAE: {min(all_mae):.2f} | Max MAE: {max(all_mae):.2f}

ERROR ANALYSIS:
  Mean Absolute Error: {np.mean(np.abs(residuals)):.2f}
  Root Mean Squared Error: {np.sqrt(np.mean(residuals**2)):.2f}
  Mean Relative Error: {100 * np.mean(np.abs(residuals) / (all_actuals + 1e-6)):.2f}%
  
PREDICTION RANGE:
  Model Predictions: [{all_predictions.min():.1f}, {all_predictions.max():.1f}]
  Actual Values: [{all_actuals.min():.1f}, {all_actuals.max():.1f}]

INTERPRETATION:
  ✓ R² > 0.5 indicates the model explains >50% of variance
  ✓ MAE < 5-6 means predictions are within 5-6 SPAD units
  ✓ Residuals should be normally distributed around 0
  
OUTPUT FILES:
  - 01_Actual_vs_Predicted.png: Scatter plot of predictions
  - 02_Residual_Plot.png: Residual analysis
  - 03_Error_Distribution.png: Error histogram
  - 04_Per_Fold_Performance.png: CV performance by fold
  - predictions.csv: Detailed predictions
  - model.pkl: Trained model (for inference)

=============================================
Generated: June 2026
=============================================
"""

with open(os.path.join(OUTPUT_DIR, 'REPORT.txt'), 'w', encoding='utf-8') as f:
    f.write(report)

print("✓ Report saved: REPORT.txt")
print("\n" + report)
