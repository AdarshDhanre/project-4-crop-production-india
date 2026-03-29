"""
Model Module for Crop Production Prediction
Trains and manages multiple ML models
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle
import warnings
warnings.filterwarnings('ignore')


class CropProductionModel:
    """Class for building and training crop production prediction models"""
    
    def __init__(self, X_train, X_test, y_train, y_test):
        """
        Initialize with train-test data
        
        Parameters:
        -----------
        X_train, X_test : array-like
            Features for training and testing
        y_train, y_test : array-like
            Target values for training and testing
        """
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.models = {}
        self.results = {}
        
    def train_linear_regression(self):
        """Train Linear Regression model"""
        print("\n🔵 Training Linear Regression...")
        
        model = LinearRegression()
        model.fit(self.X_train, self.y_train)
        
        self.models['Linear Regression'] = model
        self._evaluate_model('Linear Regression', model)
        
        return model
    
    def train_decision_tree(self, max_depth=10):
        """Train Decision Tree Regressor"""
        print("\n🌳 Training Decision Tree...")
        
        model = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
        model.fit(self.X_train, self.y_train)
        
        self.models['Decision Tree'] = model
        self._evaluate_model('Decision Tree', model)
        
        return model
    
    def train_random_forest(self, n_estimators=100, max_depth=15):
        """Train Random Forest Regressor"""
        print("\n🌲 Training Random Forest...")
        
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1
        )
        model.fit(self.X_train, self.y_train)
        
        self.models['Random Forest'] = model
        self._evaluate_model('Random Forest', model)
        
        return model
    
    def train_gradient_boosting(self, n_estimators=100, learning_rate=0.1):
        """Train Gradient Boosting Regressor"""
        print("\n📈 Training Gradient Boosting...")
        
        model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=42
        )
        model.fit(self.X_train, self.y_train)
        
        self.models['Gradient Boosting'] = model
        self._evaluate_model('Gradient Boosting', model)
        
        return model
    
    def train_svr(self, kernel='rbf', C=100):
        """Train Support Vector Regressor"""
        print("\n🎯 Training SVR...")
        
        model = SVR(kernel=kernel, C=C)
        model.fit(self.X_train, self.y_train)
        
        self.models['SVR'] = model
        self._evaluate_model('SVR', model)
        
        return model
    
    def _evaluate_model(self, name, model):
        """Evaluate model and store results"""
        
        # Predictions
        y_train_pred = model.predict(self.X_train)
        y_test_pred = model.predict(self.X_test)
        
        # Metrics
        train_r2 = r2_score(self.y_train, y_train_pred)
        test_r2 = r2_score(self.y_test, y_test_pred)
        
        train_rmse = np.sqrt(mean_squared_error(self.y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(self.y_test, y_test_pred))
        
        train_mae = mean_absolute_error(self.y_train, y_train_pred)
        test_mae = mean_absolute_error(self.y_test, y_test_pred)
        
        # Cross validation
        cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5, scoring='r2')
        
        # Store results
        self.results[name] = {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
        
        # Print results
        print(f"\n  {'Metric':<20} {'Train':<15} {'Test':<15}")
        print(f"  {'-'*50}")
        print(f"  {'R² Score':<20} {train_r2:<15.4f} {test_r2:<15.4f}")
        print(f"  {'RMSE':<20} {train_rmse:<15.4f} {test_rmse:<15.4f}")
        print(f"  {'MAE':<20} {train_mae:<15.4f} {test_mae:<15.4f}")
        print(f"  {'CV Mean':<20} {cv_scores.mean():<15.4f}")
        
    def get_best_model(self):
        """Get best model based on test R² score"""
        if not self.models:
            print("❌ No models trained yet!")
            return None
        
        best_name = max(self.results, key=lambda x: self.results[x]['test_r2'])
        best_model = self.models[best_name]
        best_score = self.results[best_name]['test_r2']
        
        print(f"\n🏆 Best Model: {best_name} (R² = {best_score:.4f})")
        
        return best_name, best_model, best_score
    
    def get_all_results(self):
        """Get results of all trained models as DataFrame"""
        df_results = pd.DataFrame(self.results).T
        return df_results.sort_values('test_r2', ascending=False)
    
    def save_model(self, model_name, filepath):
        """Save trained model to file"""
        if model_name not in self.models:
            print(f"❌ Model '{model_name}' not found!")
            return
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.models[model_name], f)
        print(f"💾 Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load model from file"""
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        print(f"📂 Model loaded from {filepath}")
        return model
    
    def predict(self, model_name, X):
        """Make predictions using trained model"""
        if model_name not in self.models:
            print(f"❌ Model '{model_name}' not found!")
            return None
        
        predictions = self.models[model_name].predict(X)
        return predictions
    
    def feature_importance(self, model_name):
        """Get feature importance for tree-based models"""
        if model_name not in self.models:
            print(f"❌ Model '{model_name}' not found!")
            return None
        
        model = self.models[model_name]
        
        # Check if model has feature_importance_ attribute
        if not hasattr(model, 'feature_importances_'):
            print(f"⚠️ Model '{model_name}' doesn't support feature importance!")
            return None
        
        importance = model.feature_importances_
        return importance


def train_all_models(X_train, X_test, y_train, y_test):
    """
    Train all models and compare performance
    
    Parameters:
    -----------
    X_train, X_test : array-like
        Training and testing features
    y_train, y_test : array-like
        Training and testing targets
    
    Returns:
    --------
    CropProductionModel
        Trained model object with all models
    """
    
    print("\n" + "="*60)
    print("🚀 TRAINING ALL MODELS")
    print("="*60)
    
    # Initialize model container
    model_container = CropProductionModel(X_train, X_test, y_train, y_test)
    
    # Train all models
    model_container.train_linear_regression()
    model_container.train_decision_tree()
    model_container.train_random_forest()
    model_container.train_gradient_boosting()
    model_container.train_svr()
    
    # Get results summary
    print("\n" + "="*60)
    print("📊 MODEL COMPARISON SUMMARY")
    print("="*60)
    results_df = model_container.get_all_results()
    print(results_df)
    
    # Get best model
    best_name, best_model, best_score = model_container.get_best_model()
    
    return model_container, best_name, best_model


if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import make_regression
    X, y = make_regression(n_samples=100, n_features=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models, best_name, best_model = train_all_models(X_train, X_test, y_train, y_test)
