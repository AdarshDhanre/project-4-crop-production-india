import pandas as pd
import os

def consolidate_data():
    """
    Consolidates fragmented CSV files in data/ into a single agriculture_data.csv
    """
    print("🔄 Consolidating data fragments...")
    
    # Paths
    data_dir = 'data'
    f1 = os.path.join(data_dir, 'datafile (1).csv')
    f2 = os.path.join(data_dir, 'datafile (2).csv')
    f3 = os.path.join(data_dir, 'datafile (3).csv')
    
    # Load data
    df1 = pd.read_csv(f1)
    df2 = pd.read_csv(f2)
    df3 = pd.read_csv(f3)
    
    # Normalize crop names for merging
    df1['Crop'] = df1['Crop'].str.strip().str.upper()
    df2['Crop'] = df2['Crop             '].str.strip().str.upper() # Note the extra spaces in column name
    df3['Crop'] = df3['Crop'].str.strip().str.upper()
    
    # Merge df1 (Cost/Yield) with df3 (Variety/Season/Zone)
    # We'll take the first variety for each crop to avoid row explosion, 
    # or keep all varieties if the user wants variety-level prediction.
    # Given the description, Variety is a feature.
    
    # Sample variety/season info from df3
    df3_unique = df3.drop_duplicates(subset=['Crop'])
    
    # Merge df1 and df3_unique
    combined = pd.merge(df1, df3_unique[['Crop', 'Variety', 'Season/ duration in days', 'Recommended Zone']], on='Crop', how='left')
    
    # Merge with df2 (Production and Area)
    # We use the latest columns: Area 2010-11 and Production 2010-11
    df2_subset = df2[['Crop', 'Area 2010-11', 'Production 2010-11']]
    combined = pd.merge(combined, df2_subset, on='Crop', how='left')
    
    # Clean up column names to match user's description
    combined.rename(columns={
        'Yield (Quintal/ Hectare) ': 'Yield',
        'Cost of Production (`/Quintal) C2': 'Cost',
        'Area 2010-11': 'Quantity',
        'Production 2010-11': 'Production',
        'Season/ duration in days': 'Season'
    }, inplace=True)
    
    # Add a 'Unit' column as per description (Tons)
    combined['Unit'] = 'Tons'
    
    # Fill remaining NaNs (simple imputation for this consolidation step)
    combined['Variety'] = combined['Variety'].fillna('Generic')
    combined['Season'] = combined['Season'].fillna('Medium')
    combined['Recommended Zone'] = combined['Recommended Zone'].fillna('General')
    combined['Quantity'] = combined['Quantity'].fillna(combined['Quantity'].median())
    combined['Production'] = combined['Production'].fillna(combined['Production'].median())
    
    # Final CSV according to user's specified columns
    output_cols = ['Crop', 'Variety', 'State', 'Quantity', 'Production', 'Season', 'Unit', 'Cost', 'Recommended Zone']
    
    # Check which columns exist
    final_df = combined[[c for c in output_cols if c in combined.columns]]
    
    # Save the file
    output_path = os.path.join(data_dir, 'agriculture_data.csv')
    final_df.to_csv(output_path, index=False)
    
    print(f"✅ Data consolidation complete! Saved to {output_path}")
    print(f"📊 Summary: {len(final_df)} rows, {len(final_df.columns)} columns")
    print(f"📋 Columns: {final_df.columns.tolist()}")

if __name__ == "__main__":
    consolidate_data()
