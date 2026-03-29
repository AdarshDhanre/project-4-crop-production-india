"""
Data Preprocessing Module for Crop Production Prediction
Handles loading, cleaning, and feature engineering of agriculture data
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')


class DataPreprocessor:
    """Class for preprocessing agriculture dataset"""
    
    def __init__(self, filepath):
        """Initialize with dataset filepath"""
        self.filepath = filepath
        self.df = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def load_data(self):
        """Load CSV file"""
        print("📂 Loading data...")
        self.df = pd.read_csv(self.filepath)
        print(f"✅ Data loaded: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
        return self.df
    
    def explore_data(self):
        """Explore dataset structure and content"""
        print("\n" + "="*50)
        print("📊 DATASET EXPLORATION")
        print("="*50)
        
        print(f"\n📋 Dataset Shape: {self.df.shape}")
        print(f"\n📝 Column Names and Types:")
        print(self.df.dtypes)
        
        print(f"\n❓ Missing Values:")
        missing = self.df.isnull().sum()
        print(missing[missing > 0] if missing.sum() > 0 else "No missing values!")
        
        print(f"\n📈 Statistical Summary:")
        print(self.df.describe())
        
        print(f"\n🏷️ Categorical Columns:")
        categorical = self.df.select_dtypes(include=['object']).columns
        for col in categorical:
            print(f"  {col}: {self.df[col].nunique()} unique values")
            
    def handle_missing_values(self, strategy='drop'):
        """Handle missing values"""
        print("\n🧹 Handling missing values...")
        
        missing_count = self.df.isnull().sum().sum()
        if missing_count == 0:
            print("✅ No missing values found!")
            return self.df
        
        if strategy == 'drop':
            self.df = self.df.dropna()
            print(f"✅ Dropped rows with missing values. Shape: {self.df.shape}")
        elif strategy == 'fill':
            # Fill numerical with median, categorical with mode
            for col in self.df.columns:
                if self.df[col].dtype in ['float64', 'int64']:
                    self.df[col] = self.df[col].fillna(self.df[col].median())
                else:
                    self.df[col] = self.df[col].fillna(self.df[col].mode()[0])
            print(f"✅ Filled missing values. Shape: {self.df.shape}")
        
        return self.df
    
    def remove_duplicates(self):
        """Remove duplicate rows"""
        print("\n🔄 Checking for duplicates...")
        duplicates = self.df.duplicated().sum()
        if duplicates > 0:
            self.df = self.df.drop_duplicates()
            print(f"✅ Removed {duplicates} duplicate rows")
        else:
            print("✅ No duplicates found!")
        return self.df
    
    def encode_categorical(self):
        """Encode categorical variables"""
        print("\n🔡 Encoding categorical variables...")
        
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            le = LabelEncoder()
            self.df[col] = le.fit_transform(self.df[col])
            self.label_encoders[col] = le
            print(f"  ✅ Encoded '{col}'")
        
        return self.df
    
    def feature_engineering(self):
        """Create new features from existing ones"""
        print("\n⚙️ Feature Engineering...")
        
        # Example: If you have year column, create year-based features
        if 'Year' in self.df.columns or 'year' in self.df.columns:
            year_col = 'Year' if 'Year' in self.df.columns else 'year'
            self.df['Year_Encoded'] = self.df[year_col] - self.df[year_col].min()
            print("  ✅ Created Year_Encoded feature")
        
        # Example: If you have Season, create dummy variables
        if 'Season' in self.df.columns:
            season_dummies = pd.get_dummies(self.df['Season'], prefix='Season')
            self.df = pd.concat([self.df, season_dummies], axis=1)
            print("  ✅ Created Season dummy variables")
        
        return self.df
    
    def scale_features(self, columns=None):
        """Scale numerical features to 0-1 range"""
        print("\n📏 Scaling features...")
        
        if columns is None:
            columns = self.df.select_dtypes(include=['float64', 'int64']).columns
        
        self.df[columns] = self.scaler.fit_transform(self.df[columns])
        print(f"  ✅ Scaled {len(columns)} features")
        
        return self.df
    
    def remove_outliers(self, columns=None, method='iqr'):
        """Remove outliers using IQR method"""
        print("\n🎯 Removing outliers...")
        
        if columns is None:
            columns = self.df.select_dtypes(include=['float64', 'int64']).columns
        
        initial_rows = len(self.df)
        
        for col in columns:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            self.df = self.df[(self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)]
        
        removed = initial_rows - len(self.df)
        print(f"  ✅ Removed {removed} outlier rows")
        
        return self.df
    
    def get_processed_data(self):
        """Return processed dataframe"""
        return self.df
    
    def save_processed_data(self, output_path):
        """Save processed data to CSV"""
        self.df.to_csv(output_path, index=False)
        print(f"\n💾 Processed data saved to {output_path}")


def preprocess_pipeline(filepath, output_path=None):
    """
    Complete preprocessing pipeline
    
    Parameters:
    -----------
    filepath : str
        Path to raw CSV file
    output_path : str
        Path to save processed data (optional)
    
    Returns:
    --------
    pd.DataFrame
        Preprocessed dataframe
    """
    
    print("\n" + "="*60)
    print("🚀 STARTING DATA PREPROCESSING PIPELINE")
    print("="*60)
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(filepath)
    
    # Execute pipeline
    preprocessor.load_data()
    preprocessor.explore_data()
    preprocessor.handle_missing_values(strategy='drop')
    preprocessor.remove_duplicates()
    preprocessor.encode_categorical()
    preprocessor.feature_engineering()
    preprocessor.remove_outliers()
    preprocessor.scale_features()
    
    # Get processed data
    processed_df = preprocessor.get_processed_data()
    
    # Save if output path provided
    if output_path:
        preprocessor.save_processed_data(output_path)
    
    print("\n" + "="*60)
    print("✅ PREPROCESSING COMPLETE!")
    print("="*60)
    
    return processed_df


if __name__ == "__main__":
    # Example usage
    filepath = '../data/agriculture_data.csv'
    processed_data = preprocess_pipeline(filepath, output_path='../data/processed_data.csv')
    print("\n✨ Data is ready for modeling!")
