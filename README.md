# 🔥 Fire Risk Predictor

Machine learning system for predicting fire risk using historical fire occurrence data. This project trains multiple models (Random Forest, MLP, XGBoost) to predict fire incidents with optimized classification thresholds.

## Overview

The fire risk predictor uses historical fire data (`sisam_focos_2003.csv`) to train three regression models that output fire probability predictions (0-1 range). These probabilities are converted to binary classifications using optimized threshold values.

### Models

- **Random Forest Regressor** - Ensemble tree-based model
- **Multi-Layer Perceptron (MLP)** - Neural network with StandardScaler preprocessing
- **XGBoost Regressor** - Gradient boosting model

### Training Process

1. **Data preprocessing**: Drop unnecessary columns, handle null values
2. **Balanced sampling**: 125k fire samples + 125k non-fire samples (250k total)
3. **K-Fold cross-validation**: Train with K=3 and K=5 folds
4. **Threshold optimization**: Grid search (0.1-0.9) to maximize F1 score
5. **Evaluation**: Test on separate 50k sample set

## Project Structure

```
fire-risk-predictor/
├── FireRiskPredictor.ipynb    # Main training notebook
├── .github/
│   └── workflows/
│       └── sync-to-r2.yml      # GitHub Actions workflow for R2 sync
├── sisam_focos_2003.csv        # Training dataset (not in git)
├── RF.pkl                      # Trained Random Forest model (not in git)
├── MLP.pkl                     # Trained MLP model (not in git)
├── XGBoost.pkl                 # Trained XGBoost model (not in git)
└── README.md
```

## Data Storage

Large files (datasets and trained models) are stored in **Cloudflare R2** and synced via GitHub Actions:

- **Bucket**: `fire-risk-predictor`
- **Sync workflow**: `.github/workflows/sync-to-r2.yml`
- **Source**: Google Drive folder `fire-risk-predictor`

See [.github/workflows/README.md](.github/workflows/README.md) for sync setup instructions.

## Requirements

```bash
pip install pandas matplotlib scikit-learn xgboost joblib tqdm seaborn
```

## Usage

### Training Models

Run the Jupyter notebook `FireRiskPredictor.ipynb` cells sequentially:

1. Install dependencies
2. Load and preprocess data
3. Train models with K-Fold cross-validation
4. Optimize classification thresholds
5. Evaluate on test set
6. Process cluster-specific predictions

### Using Pre-trained Models

```python
import joblib
import pandas as pd

# Load model
rf_model = joblib.load('RF.pkl')

# Prepare data (must match training features)
X_new = df.drop(columns=['incendio', 'datahora', 'longitude', 'latitude'])

# Predict probabilities
probabilities = rf_model.predict(X_new)

# Apply threshold (use optimized value from training)
predictions = (probabilities >= 0.5).astype(int)
```

## Model Performance

Models are evaluated using:
- **Mean Squared Error (MSE)** - Regression performance
- **R² Score** - Explained variance
- **Accuracy, Precision, Recall, F1 Score** - Classification metrics
- **Confusion Matrix** - True/false positives and negatives

## Future Development

This project is being transformed into a scalable web application:

- **Frontend**: Vite SPA for file uploads and visualization
- **Backend**: FastAPI for model training and inference
- **Storage**: Cloudflare R2 for datasets and models
- **Features**: 
  - User-uploaded CSV processing
  - Async model training with progress tracking
  - Real-time predictions with threshold optimization
  - Batch processing for cluster files

## License

[Add your license here]

## Contributors

[Add contributors here]
