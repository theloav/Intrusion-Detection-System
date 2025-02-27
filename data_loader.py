import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_data(file_paths):
    """Load and combine datasets."""
    combined_df = pd.concat([pd.read_csv(file) for file in file_paths], ignore_index=True)
    return combined_df

def preprocess_data(df):
    """Preprocess the DataFrame."""
    # Remove leading and trailing spaces from column names
    df.columns = df.columns.str.strip()

    # Check for null values
    print("Null values count:\n", df.isnull().sum())
    
    # Label Encoding on the 'Label' column
    le = LabelEncoder()
    df['Label'] = le.fit_transform(df['Label'])
    
    return df, le
