import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def clean_dataframe(df):
    """
    Cleans the DataFrame by replacing 'N/S' with NA and dropping columns with any missing values.
    Returns the cleaned DataFrame along with the number of columns removed.
    """
    # Replace 'N/S', 'non', or any other specific ill-defined values with NaN
    df.replace(["N/S", "non"], [None, None], inplace=True)

    # Get the initial number of columns
    initial_cols = df.shape[1]

    # Remove columns that have any NaN values (i.e., originally had 'N/S' or 'non')
    df.dropna(axis=1, inplace=True)

    # Drop columns that contain any missing values
    df.dropna(axis=1, how="any", inplace=True)

    # Calculate the number of columns removed
    removed_col_count = initial_cols - df.shape[1]

    return df, removed_col_count

def standardize_data(X, fit_scaler=True, scaler_standard=None, scaler_minmax=None):
    """
    Normalizes the data by combining autoscaling and rescaling to [0, 1].
    Returns scaled data along with scalers for consistent preprocessing.

    Parameters:
        X: Feature matrix to be normalized.
        fit_scaler: Whether to fit new scalers (default=True).
        scaler_standard: Existing StandardScaler to reuse (default=None).
        scaler_minmax: Existing MinMaxScaler to reuse (default=None).

    Returns:
        X_normalized: Normalized data.
        scaler_standard: Fitted or reused StandardScaler.
        scaler_minmax: Fitted or reused MinMaxScaler.
    """
    if fit_scaler:
        scaler_standard = StandardScaler()
        X_scaled = scaler_standard.fit_transform(X)
        scaler_minmax = MinMaxScaler()
        X_normalized = scaler_minmax.fit_transform(X_scaled)
    else:
        # Use existing scalers to transform data
        X_scaled = scaler_standard.transform(X)
        X_normalized = scaler_minmax.transform(X_scaled)

    return X_normalized, scaler_standard, scaler_minmax

def drop_columns(df):
    """
    Drops columns from a DataFrame if they exist.
    """
    cols_to_drop = [
        "Formula",
        "formula",
        "Structure",
        "Structure type",
        "structure_type",
        "A",
        "B",
        "Entry",
    ]
    for col in cols_to_drop:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)

    return df

def prepare_standarlize_X_block_(csv_file_path):
    """
    Reads the data, cleans the DataFrame, removes specific columns, and normalizes the data by default.
    Returns scalers for use with validation data.
    """
    df = pd.read_csv(csv_file_path)

    # Check the shape of the dataset
    print(f"Loaded data shape: {df.shape}")

    # Drop unnecessary columns
    X_df = drop_columns(df)
    print(f"Data shape after dropping columns: {X_df.shape}")

    # Clean the DataFrame
    X_df, removed_col_count = clean_dataframe(X_df)
    print(f"Removed {removed_col_count} columns with invalid values.")
    print(f"Data shape after cleaning: {X_df.shape}")

    # Normalize the data and retrieve scalers
    X, scaler_standard, scaler_minmax = standardize_data(X_df)
    print(f"Shape of normalized data: {X.shape}")

    # Get the list of column names
    columns = X_df.columns.tolist()

    return X_df, X, columns, scaler_standard, scaler_minmax

def preprocess_validation_data(csv_file_path, scaler_standard, scaler_minmax):
    """
    Preprocesses validation data using the same scalers as the training data.
    
    Parameters:
        csv_file_path: Path to the validation dataset.
        scaler_standard: Fitted StandardScaler from training data.
        scaler_minmax: Fitted MinMaxScaler from training data.
    
    Returns:
        X_df: Cleaned and ordered validation DataFrame.
        X: Normalized validation feature matrix.
    """
    df = pd.read_csv(csv_file_path)

    # Drop unnecessary columns
    X_df = drop_columns(df)
    print(f"Validation data shape after dropping columns: {X_df.shape}")

    # Clean the DataFrame
    X_df, removed_col_count = clean_dataframe(X_df)
    print(f"Removed {removed_col_count} columns with invalid values in validation data.")
    print(f"Validation data shape after cleaning: {X_df.shape}")

    # Ensure validation columns match training columns
    training_columns = scaler_standard.feature_names_in_  # Retrieve training feature names
    if list(X_df.columns) != list(training_columns):
        print("Reordering validation columns to match training data...")
        X_df = X_df[training_columns]

    # Normalize validation data using the provided scalers
    X, _, _ = standardize_data(
        X_df, fit_scaler=False, scaler_standard=scaler_standard, scaler_minmax=scaler_minmax
    )
    print(f"Shape of normalized validation data: {X.shape}")

    return X_df, X