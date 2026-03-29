"""
Project 4: Crop Production Prediction in India
Main Script - Complete Pipeline
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import sys
import os

# Import custom modules
from src.data_preprocessing import DataPreprocessor, preprocess_pipeline
from src.model import train_all_models
from src.evaluation import ModelEvaluator, compare_models, plot_model_comparison, generate_evaluation_report

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


def main():
    """Main pipeline execution"""
    
    print("\n" + "="*70)
    print("🌾 PROJECT 4: CROP PRODUCTION PREDICTION IN INDIA")
    print("="*70)
    
    # ============================================================
    # STEP 1: DATA LOADING & PREPROCESSING
    # ============================================================
    print("\n\n📊 STEP 1: DATA LOADING & PREPROCESSING")
    print("-"*70)
    
    # Load and preprocess data
    filepath = 'data/agriculture_data.csv'
    processed_df = preprocess_pipeline(filepath, output_path='data/processed_data.csv')
    
    # ============================================================
    # STEP 2: DATA PREPARATION FOR MODELING
    # ============================================================
    print("\n\n🔧 STEP 2: DATA PREPARATION FOR MODELING")
    print("-"*70)
    
    # Separate features and target
    # Adjust column names based on your actual dataset
    # Common columns might be: 'Production', 'Quantity', 'Cost', etc.
    
    # List all columns to decide which to use
    print(f"\n📋 Available columns in dataset:")
    print(processed_df.columns.tolist())
    
    # Example: Using numeric columns and excluding the target
    # Modify this based on your actual dataset structure
    target_col = 'Production'  # Change if your target column is different
    
    # Get feature columns (exclude target and non-numeric)
    feature_cols = processed_df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in feature_cols:
        feature_cols.remove(target_col)
    
    print(f"\n🎯 Target column: {target_col}")
    print(f"📌 Feature columns: {feature_cols}")
    
    # Prepare X and y
    X = processed_df[feature_cols].values
    y = processed_df[target_col].values
    
    print(f"\n✅ Features shape: {X.shape}")
    print(f"✅ Target shape: {y.shape}")
    
    # ============================================================
    # STEP 3: TRAIN-TEST SPLIT
    # ============================================================
    print("\n\n📉 STEP 3: TRAIN-TEST SPLIT")
    print("-"*70)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"✅ Training set size: {X_train.shape[0]}")
    print(f"✅ Testing set size: {X_test.shape[0]}")
    print(f"✅ Training-Testing split: 80-20")
    
    # ============================================================
    # STEP 4: MODEL TRAINING
    # ============================================================
    print("\n\n🤖 STEP 4: MODEL TRAINING")
    print("-"*70)
    
    # Train all models
    model_container, best_model_name, best_model = train_all_models(
        X_train, X_test, y_train, y_test
    )
    
    # ============================================================
    # STEP 5: MODEL EVALUATION & COMPARISON
    # ============================================================
    print("\n\n📈 STEP 5: MODEL EVALUATION & COMPARISON")
    print("-"*70)
    
    # Get all models
    models_dict = model_container.models
    
    # Compare models
    comparison_df = compare_models(models_dict, X_test, y_test)
    
    # Save comparison
    comparison_df.to_csv('results/model_comparison.csv')
    print(f"\n💾 Comparison saved to 'results/model_comparison.csv'")
    
    # ============================================================
    # STEP 6: BEST MODEL DETAILED EVALUATION
    # ============================================================
    print("\n\n🏆 STEP 6: BEST MODEL DETAILED EVALUATION")
    print("-"*70)
    
    y_pred_best = best_model.predict(X_test)
    best_evaluator = ModelEvaluator(y_test, y_pred_best, best_model_name)
    best_metrics = best_evaluator.calculate_metrics()
    best_evaluator.print_metrics()
    
    # ============================================================
    # STEP 7: VISUALIZATION
    # ============================================================
    print("\n\n📊 STEP 7: CREATING VISUALIZATIONS")
    print("-"*70)
    
    # Create results directory if doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Plot actual vs predicted
    best_evaluator.plot_actual_vs_predicted('results/actual_vs_predicted.png')
    
    # Plot residuals
    best_evaluator.plot_residuals('results/residuals_plot.png')
    
    # Plot error distribution
    best_evaluator.plot_error_distribution('results/error_distribution.png')
    
    # Plot model comparison
    plot_model_comparison(comparison_df, 'results/model_comparison.png')
    
    # Feature importance (if available)
    if best_model_name in ['Random Forest', 'Gradient Boosting', 'Decision Tree']:
        try:
            from src.evaluation import plot_feature_importance
            plot_feature_importance(best_model, feature_cols, 'results/feature_importance.png')
        except:
            print("⚠️ Could not plot feature importance")
    
    # ============================================================
    # STEP 8: SAVE BEST MODEL
    # ============================================================
    print("\n\n💾 STEP 8: SAVING BEST MODEL")
    print("-"*70)
    
    model_container.save_model(best_model_name, f'models/{best_model_name.lower().replace(" ", "_")}.pkl')
    
    # ============================================================
    # STEP 9: GENERATE REPORT
    # ============================================================
    print("\n\n📄 STEP 9: GENERATING EVALUATION REPORT")
    print("-"*70)
    
    os.makedirs('results', exist_ok=True)
    generate_evaluation_report(
        best_model, X_test, y_test, best_model_name,
        output_file='results/evaluation_report.txt'
    )
    
    # ============================================================
    # FINAL SUMMARY
    # ============================================================
    print("\n\n" + "="*70)
    print("✅ PIPELINE EXECUTION COMPLETED!")
    print("="*70)
    
    summary = f"""
📊 FINAL SUMMARY:
{'-'*70}
Best Model: {best_model_name}
R² Score: {best_metrics['R²']:.4f}
RMSE: {best_metrics['RMSE']:.4f}
MAE: {best_metrics['MAE']:.4f}

📁 Output Files Generated:
  ✅ results/model_comparison.csv - Model performance comparison
  ✅ results/actual_vs_predicted.png - Prediction visualization
  ✅ results/residuals_plot.png - Residual analysis
  ✅ results/error_distribution.png - Error distribution
  ✅ results/model_comparison.png - Model comparison plot
  ✅ results/feature_importance.png - Feature importance plot
  ✅ results/evaluation_report.txt - Detailed report
  ✅ models/{best_model_name.lower().replace(' ', '_')}.pkl - Saved model

📈 Next Steps:
  1. Review the visualizations in 'results/' folder
  2. Check the evaluation report for detailed analysis
  3. Deploy the best model for predictions
  4. Document findings in the project report

{'-'*70}
Thank you for using this pipeline! 🚀
"""
    
    print(summary)
    
    return model_container, best_model_name, best_model


if __name__ == "__main__":
    try:
        model_container, best_model_name, best_model = main()
        print("\n✨ All processes completed successfully!")
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
