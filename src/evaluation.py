"""
Evaluation Module for Crop Production Prediction
Evaluation metrics and visualization functions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class ModelEvaluator:
    """Class for evaluating model performance"""
    
    def __init__(self, y_true, y_pred, model_name="Model"):
        """
        Initialize evaluator
        
        Parameters:
        -----------
        y_true : array-like
            Actual values
        y_pred : array-like
            Predicted values
        model_name : str
            Name of the model
        """
        self.y_true = y_true
        self.y_pred = y_pred
        self.model_name = model_name
        self.metrics = {}
        
    def calculate_metrics(self):
        """Calculate all evaluation metrics"""
        
        # Mean Squared Error
        mse = mean_squared_error(self.y_true, self.y_pred)
        rmse = np.sqrt(mse)
        
        # Mean Absolute Error
        mae = mean_absolute_error(self.y_true, self.y_pred)
        
        # Mean Absolute Percentage Error
        mape = mean_absolute_percentage_error(self.y_true, self.y_pred)
        
        # R² Score
        r2 = r2_score(self.y_true, self.y_pred)
        
        # Adjusted R²
        n = len(self.y_true)
        k = 1  # number of predictors
        adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1)
        
        # Store metrics
        self.metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'R²': r2,
            'Adjusted R²': adjusted_r2
        }
        
        return self.metrics
    
    def print_metrics(self):
        """Print evaluation metrics"""
        if not self.metrics:
            self.calculate_metrics()
        
        print(f"\n📊 {self.model_name} - Evaluation Metrics")
        print("="*50)
        for metric_name, value in self.metrics.items():
            print(f"{metric_name:<20}: {value:.4f}")
        print("="*50)
    
    def plot_actual_vs_predicted(self, save_path=None):
        """Plot actual vs predicted values"""
        if not self.metrics:
            self.calculate_metrics()
        
        plt.figure(figsize=(10, 6))
        plt.scatter(self.y_true, self.y_pred, alpha=0.6, edgecolors='k')
        
        # Perfect prediction line
        min_val = min(self.y_true.min(), self.y_pred.min())
        max_val = max(self.y_true.max(), self.y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        plt.xlabel('Actual Production', fontsize=12)
        plt.ylabel('Predicted Production', fontsize=12)
        plt.title(f'{self.model_name} - Actual vs Predicted', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Plot saved to {save_path}")
        
        # plt.show()
    
    def plot_residuals(self, save_path=None):
        """Plot residuals"""
        residuals = self.y_true - self.y_pred
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Residuals vs Predicted
        axes[0].scatter(self.y_pred, residuals, alpha=0.6, edgecolors='k')
        axes[0].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[0].set_xlabel('Predicted Values', fontsize=11)
        axes[0].set_ylabel('Residuals', fontsize=11)
        axes[0].set_title('Residuals vs Predicted Values', fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # Residuals histogram
        axes[1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        axes[1].set_xlabel('Residuals', fontsize=11)
        axes[1].set_ylabel('Frequency', fontsize=11)
        axes[1].set_title('Distribution of Residuals', fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Plot saved to {save_path}")
        
        # plt.show()
    
    def plot_error_distribution(self, save_path=None):
        """Plot absolute error distribution"""
        errors = np.abs(self.y_true - self.y_pred)
        
        plt.figure(figsize=(10, 6))
        plt.hist(errors, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
        plt.xlabel('Absolute Error', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title(f'{self.model_name} - Absolute Error Distribution', fontsize=14, fontweight='bold')
        plt.axvline(errors.mean(), color='r', linestyle='--', lw=2, label=f'Mean Error: {errors.mean():.2f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Plot saved to {save_path}")
        
        # plt.show()


def compare_models(models_dict, X_test, y_test):
    """
    Compare multiple models
    
    Parameters:
    -----------
    models_dict : dict
        Dictionary of {model_name: model_object}
    X_test : array-like
        Test features
    y_test : array-like
        Test targets
    
    Returns:
    --------
    pd.DataFrame
        Comparison table
    """
    
    print("\n" + "="*60)
    print("📊 MODEL COMPARISON")
    print("="*60)
    
    results = []
    
    for model_name, model in models_dict.items():
        y_pred = model.predict(X_test)
        evaluator = ModelEvaluator(y_test, y_pred, model_name)
        metrics = evaluator.calculate_metrics()
        metrics['Model'] = model_name
        results.append(metrics)
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(results)
    comparison_df = comparison_df.set_index('Model')
    comparison_df = comparison_df.sort_values('R²', ascending=False)
    
    print("\n" + comparison_df.to_string())
    
    return comparison_df


def plot_model_comparison(comparison_df, save_path=None):
    """Plot model comparison"""
    
    metrics_to_plot = ['R²', 'RMSE', 'MAE']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, metric in enumerate(metrics_to_plot):
        if metric in comparison_df.columns:
            comparison_df[metric].plot(kind='bar', ax=axes[idx], color='steelblue', edgecolor='black')
            axes[idx].set_title(f'{metric} Comparison', fontweight='bold')
            axes[idx].set_ylabel(metric, fontsize=11)
            axes[idx].set_xlabel('Model', fontsize=11)
            axes[idx].tick_params(axis='x', rotation=45)
            axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Plot saved to {save_path}")
    
    # plt.show()


def plot_feature_importance(model, feature_names, save_path=None):
    """Plot feature importance for tree-based models"""
    
    if not hasattr(model, 'feature_importances_'):
        print("⚠️ Model doesn't support feature importance!")
        return
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title('Feature Importance', fontweight='bold', fontsize=14)
    plt.bar(range(len(importances)), importances[indices], align='center', edgecolor='black')
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.xlabel('Feature', fontsize=12)
    plt.ylabel('Importance', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Plot saved to {save_path}")
    
    # plt.show()


def generate_evaluation_report(model, X_test, y_test, model_name="Best Model", output_file=None):
    """Generate comprehensive evaluation report"""
    
    y_pred = model.predict(X_test)
    evaluator = ModelEvaluator(y_test, y_pred, model_name)
    metrics = evaluator.calculate_metrics()
    
    report = f"""
{'='*60}
{model_name} - EVALUATION REPORT
{'='*60}

PERFORMANCE METRICS:
{'-'*60}
"""
    
    for metric_name, value in metrics.items():
        report += f"{metric_name:<20}: {value:.4f}\n"
    
    report += f"""
{'-'*60}

INTERPRETATION:
- R² Score: {metrics['R²']:.4f} (Explained variance: {metrics['R²']*100:.2f}%)
- RMSE: {metrics['RMSE']:.4f} (Root Mean Squared Error)
- MAE: {metrics['MAE']:.4f} (Mean Absolute Error)
- MAPE: {metrics['MAPE']:.4f} (Mean Absolute Percentage Error)

{'='*60}
"""
    
    print(report)
    
    if output_file:
        with open(output_file, 'w') as f:
            f.write(report)
        print(f"💾 Report saved to {output_file}")
    
    return report


if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    
    X, y = make_regression(n_samples=100, n_features=5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    evaluator = ModelEvaluator(y_test, y_pred, "Random Forest")
    evaluator.calculate_metrics()
    evaluator.print_metrics()
    evaluator.plot_actual_vs_predicted()
    evaluator.plot_residuals()
