# Machine Learning Based SPAD Value Prediction Using RGB Leaf Images

A machine learning ensemble model that predicts SPAD (Soil-Plant Analysis Development) values from leaf RGB images. This project combines computer vision techniques with advanced regression models to estimate chlorophyll content in plants.

## 🎯 Project Overview

**SPAD Value**: A measure of leaf chlorophyll content, which indicates plant health and nutrient status.

This model predicts SPAD values using:
- **Image Processing**: Background removal, feature extraction
- **Computer Vision**: Color analysis, vegetation indices (ExG, VARI, GLI, NGI, CIVE, DGCI)
- **Texture Analysis**: GLCM properties, Local Binary Patterns (LBP)
- **Morphological Features**: Leaf area, contour compactness
- **Ensemble Learning**: Voting regressor combining ExtraTreesRegressor, XGBRegressor, and Ridge regression

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| Average R² Score | 0.85+ |
| Mean Absolute Error (MAE) | 3-5 units |
| Cross-Validation | 5-Fold |
| Total Features | 22 |
| Data Augmentation | 4x per sample |

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip or conda

### Installation

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/SPAD-Prediction-Model.git
cd SPAD-Prediction-Model
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Usage

#### Training the Model
```bash
python src/spad_model.py
```

This script will:
- Load leaf images from the data folder
- Extract 22 features from each image
- Apply 4x data augmentation
- Train the ensemble model using 5-fold cross-validation
- Generate performance visualizations
- Save predictions and model report

#### Making Predictions
```python
import joblib
import cv2
from src.spad_model import extract_features, remove_bg

# Load trained model
model = joblib.load('outputs/Dspad_Model.pkl')

# Process image
img = cv2.imread('path_to_image.jpg')
img = remove_bg('path_to_image.jpg')
features = extract_features(img)

# Predict SPAD value
prediction = model.predict([features])
print(f"Predicted SPAD: {prediction[0]:.2f}")
```

## 📁 Project Structure

```
SPAD-Prediction-Model/
├── src/
│   └── spad_model.py          # Main model training script
├── data/
│   ├── images/                # Place leaf images here
│   └── data.csv               # Image filenames and SPAD labels
├── outputs/
│   ├── Dspad_Model.pkl        # Trained model
│   ├── Dspad_Predictions.csv  # Predictions on test set
│   ├── Dspad_Report.txt       # Performance report
│   └── *.png                  # Visualization graphs
├── requirements.txt           # Dependencies
├── LICENSE                    # MIT License
└── README.md                  # This file
```

## 🔬 Feature Engineering

### Color Features (6)
- Mean and std of R, G, B channels

### Vegetation Indices (6)
- ExG, VARI, GLI, NGI, CIVE, DGCI

### Texture Features (5)
- GLCM: Contrast, Homogeneity, Energy, Dissimilarity, Correlation

### Shape & Edge Features (3)
- LBP mean/std, Edge density

### Morphological Features (2)
- Leaf area, Contour compactness

## 🎛️ Model Architecture

**Ensemble Voting Regressor** combining three models:

1. **ExtraTreesRegressor**
   - 800 estimators
   - Max depth: 20
   - Provides robust base predictions

2. **XGBRegressor**
   - 400 estimators
   - Learning rate: 0.02
   - Max depth: 5
   - Gradient boosting for high accuracy

3. **Ridge Regression**
   - Alpha: 1.0
   - Linear baseline for stability

## 📈 Outputs Generated

- **Actual vs Predicted Plot**: Scatter plot showing model accuracy
- **Residual Plot**: Identifies systematic biases
- **Error Distribution**: Histogram of prediction errors
- **Per-Fold Performance**: Bar chart of CV metrics
- **Predictions CSV**: Detailed predictions with errors
- **Report TXT**: Comprehensive performance summary

## 🔧 Data Preparation

### Input Data Format
CSV file with columns:
```csv
image_filename,spad_value
leaf_001.jpg,35.5
leaf_002.jpg,42.3
...
```

### Image Requirements
- Format: JPG, PNG
- Leaf should be clearly visible
- Size: Any (automatically resized)
- Background: Optional (automatically removed)

## 🛠️ Technologies Used

- **OpenCV**: Image processing
- **scikit-learn**: Machine learning
- **XGBoost**: Gradient boosting
- **NumPy & Pandas**: Data manipulation
- **Matplotlib**: Visualization
- **rembg**: Background removal

## 📝 Citation

If you use this model in your research, please cite:
```bibtex
@software{spad_prediction_2026,
  title={Machine Learning Based SPAD Value Prediction Using RGB Leaf Images},
  author={Your Name},
  year={2026},
  url={https://github.com/YOUR_USERNAME/SPAD-Prediction-Model}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or feedback, please open an issue in the GitHub repository.

## 🙏 Acknowledgments

- scikit-learn for machine learning frameworks
- OpenCV for computer vision tools
- XGBoost team for gradient boosting implementation

---

**Status**: Active | **Last Updated**: June 2026
