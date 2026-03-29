"""
Project 4: Crop Production Prediction in India
Main Script - UPDATED with Correct Column Names
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("\n" + "="*70)
print("🌾 PROJECT 4: CROP PRODUCTION PREDICTION IN INDIA")
print("="*70)

# ============================================================
# STEP 1: LOAD DATA
# ============================================================
print("\n📂 STEP 1: LOADING DATA")
print("-"*70)

try:
    df = pd.read_csv('data/agriculture_data.csv')
    print(f"✅ Data loaded successfully!")
    print(f"   Shape: {df.shape[0]} rows × {df.shape[1]} columns")
except FileNotFoundError:
    print("❌ File not found! Place 'agriculture_data.csv' in 'data/' folder")
    exit()

# ============================================================
# STEP 2: EXPLORE DATA
# ============================================================
print("\n📊 STEP 2: DATA EXPLORATION")
print("-"*70)

print(f"\n📋 Column Names and Types:")
print(df.dtypes)

print(f"\n📈 Dataset Statistics:")
print(df.head(10))

print(f"\n❓ Missing Values:")
print(df.isnull().sum())

# ============================================================
# STEP 3: DATA PREPROCESSING
# ============================================================
print("\n🧹 STEP 3: DATA PREPROCESSING")
print("-"*70)

# Remove missing values
print(f"\n📊 Before cleaning: {df.shape}")
df = df.dropna()
print(f"✅ After cleaning: {df.shape}")

# Remove duplicates
duplicates = df.duplicated().sum()
if duplicates > 0:
    df = df.drop_duplicates()
    print(f"✅ Removed {duplicates} duplicates")

# ============================================================
# STEP 4: ENCODE CATEGORICAL VARIABLES
# ============================================================
print("\n🔡 STEP 4: ENCODING CATEGORICAL VARIABLES")
print("-"*70)

# Dictionary to store label encoders
label_encoders = {}

# Categorical columns
categorical_cols = ['crop', 'Variety', 'state', 'Season', 'Unit', 'Recommended Zone']

for col in categorical_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
        print(f"  ✅ Encoded '{col}'")

# ============================================================
# STEP 5: PREPARE FEATURES AND TARGET
# ============================================================
print("\n🎯 STEP 5: FEATURE SELECTION")
print("-"*70)

# Features (all columns except target)
feature_cols = [col for col in df.columns if col != 'production']
X = df[feature_cols]
y = df['production']

print(f"📌 Features: {feature_cols}")
print(f"🎯 Target: production")
print(f"✅ Features shape: {X.shape}")
print(f"✅ Target shape: {y.shape}")

# ============================================================
# STEP 6: TRAIN-TEST SPLIT
# ============================================================
print("\n📉 STEP 6: TRAIN-TEST SPLIT")
print("-"*70)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"✅ Training set: {X_train.shape[0]} samples")
print(f"✅ Testing set: {X_test.shape[0]} samples")

# ============================================================
# STEP 7: TRAIN MODELS
# ============================================================
print("\n🤖 STEP 7: TRAINING MODELS")
print("-"*70)

models = {}
results = {}

# Model 1: Linear Regression
print("\n🔵 Training Linear Regression...")
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
r2_lr = r2_score(y_test, y_pred_lr)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
models['Linear Regression'] = lr
results['Linear Regression'] = {'R²': r2_lr, 'RMSE': rmse_lr}
print(f"  R² Score: {r2_lr:.4f}")
print(f"  RMSE: {rmse_lr:.4f}")

# Model 2: Decision Tree
print("\n🌳 Training Decision Tree...")
dt = DecisionTreeRegressor(max_depth=15, random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
r2_dt = r2_score(y_test, y_pred_dt)
rmse_dt = np.sqrt(mean_squared_error(y_test, y_pred_dt))
models['Decision Tree'] = dt
results['Decision Tree'] = {'R²': r2_dt, 'RMSE': rmse_dt}
print(f"  R² Score: {r2_dt:.4f}")
print(f"  RMSE: {rmse_dt:.4f}")

# Model 3: Random Forest
print("\n🌲 Training Random Forest...")
rf = RandomForestRegressor(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
r2_rf = r2_score(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
models['Random Forest'] = rf
results['Random Forest'] = {'R²': r2_rf, 'RMSE': rmse_rf}
print(f"  R² Score: {r2_rf:.4f}")
print(f"  RMSE: {rmse_rf:.4f}")

# Model 4: Gradient Boosting
print("\n📈 Training Gradient Boosting...")
gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
gb.fit(X_train, y_train)
y_pred_gb = gb.predict(X_test)
r2_gb = r2_score(y_test, y_pred_gb)
rmse_gb = np.sqrt(mean_squared_error(y_test, y_pred_gb))
models['Gradient Boosting'] = gb
results['Gradient Boosting'] = {'R²': r2_gb, 'RMSE': rmse_gb}
print(f"  R² Score: {r2_gb:.4f}")
print(f"  RMSE: {rmse_gb:.4f}")

# Model 5: SVR
print("\n🎯 Training SVR...")
svr = SVR(kernel='rbf', C=100)
svr.fit(X_train, y_train)
y_pred_svr = svr.predict(X_test)
r2_svr = r2_score(y_test, y_pred_svr)
rmse_svr = np.sqrt(mean_squared_error(y_test, y_pred_svr))
models['SVR'] = svr
results['SVR'] = {'R²': r2_svr, 'RMSE': rmse_svr}
print(f"  R² Score: {r2_svr:.4f}")
print(f"  RMSE: {rmse_svr:.4f}")

# ============================================================
# STEP 8: MODEL COMPARISON
# ============================================================
print("\n📊 STEP 8: MODEL COMPARISON")
print("-"*70)

results_df = pd.DataFrame(results).T
results_df = results_df.sort_values('R²', ascending=False)

print("\n" + results_df.to_string())

best_model_name = results_df.index[0]
best_model = models[best_model_name]
best_r2 = results_df['R²'].iloc[0]
best_rmse = results_df['RMSE'].iloc[0]

print(f"\n🏆 BEST MODEL: {best_model_name}")
print(f"   R² Score: {best_r2:.4f}")
print(f"   RMSE: {best_rmse:.4f}")

# ============================================================
# STEP 9: VISUALIZATIONS
# ============================================================
print("\n📈 STEP 9: CREATING VISUALIZATIONS")
print("-"*70)

import os
os.makedirs('results', exist_ok=True)

# Get predictions from best model
y_pred_best = best_model.predict(X_test)

# Plot 1: Actual vs Predicted
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_best, alpha=0.6, edgecolors='k')
min_val = min(y_test.min(), y_pred_best.min())
max_val = max(y_test.max(), y_pred_best.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual Production', fontsize=12)
plt.ylabel('Predicted Production', fontsize=12)
plt.title(f'{best_model_name} - Actual vs Predicted', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('results/actual_vs_predicted.png', dpi=300, bbox_inches='tight')
print(f"  ✅ Saved: actual_vs_predicted.png")
plt.close()

# Plot 2: Residuals
residuals = y_test.values - y_pred_best
plt.figure(figsize=(10, 6))
plt.scatter(y_pred_best, residuals, alpha=0.6, edgecolors='k')
plt.axhline(y=0, color='r', linestyle='--', lw=2)
plt.xlabel('Predicted Values', fontsize=12)
plt.ylabel('Residuals', fontsize=12)
plt.title('Residuals Analysis', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.savefig('results/residuals_plot.png', dpi=300, bbox_inches='tight')
print(f"  ✅ Saved: residuals_plot.png")
plt.close()

# Plot 3: Model Comparison
plt.figure(figsize=(12, 6))
results_df['R²'].plot(kind='bar', color='steelblue', edgecolor='black', alpha=0.7)
plt.xlabel('Model', fontsize=12)
plt.ylabel('R² Score', fontsize=12)
plt.title('Model Comparison - R² Score', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('results/model_comparison.png', dpi=300, bbox_inches='tight')
print(f"  ✅ Saved: model_comparison.png")
plt.close()

# Plot 4: Feature Importance (if available)
if hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_
    indices = np.argsort(importances)[::-1][:10]  # Top 10
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(indices)), importances[indices], edgecolor='black', alpha=0.7)
    plt.xticks(range(len(indices)), [feature_cols[i] for i in indices], rotation=45, ha='right')
    plt.xlabel('Feature', fontsize=12)
    plt.ylabel('Importance', fontsize=12)
    plt.title('Top 10 Most Important Features', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('results/feature_importance.png', dpi=300, bbox_inches='tight')
    print(f"  ✅ Saved: feature_importance.png")
    plt.close()

# ============================================================
# STEP 10: FINAL SUMMARY
# ============================================================
print("\n" + "="*70)
print("✅ PIPELINE COMPLETED!")
print("="*70)

summary = f"""
📊 FINAL RESULTS:
{'-'*70}
Best Model: {best_model_name}
R² Score: {best_r2:.4f} (Explains {best_r2*100:.2f}% of variance)
RMSE: {best_rmse:.4f} (Average prediction error)
MAE: {mean_absolute_error(y_test, y_pred_best):.4f}

📁 Output Files:
  ✅ results/actual_vs_predicted.png
  ✅ results/residuals_plot.png
  ✅ results/model_comparison.png
  ✅ results/feature_importance.png (if available)

📈 Model Rankings:
"""

for idx, (model_name, row) in enumerate(results_df.iterrows(), 1):
    summary += f"{idx}. {model_name:<25} R²={row['R²']:.4f}, RMSE={row['RMSE']:.4f}\n"

summary += f"""
{'-'*70}
Next Steps:
1. Review visualizations in 'results/' folder
2. Fill weekly report with achievements
3. Push code to GitHub
4. Document findings

✨ Great job! Your model is ready for predictions! 🚀
"""

print(summary)

# Save summary to file
with open('results/summary.txt', 'w') as f:
    f.write(summary)
print(f"✅ Summary saved to results/summary.txt")
