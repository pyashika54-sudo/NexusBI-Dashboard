import logging
import pandas as pd
from typing import Union, IO

# Set up logging for the data processor module
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

def load_data(file: Union[str, IO]) -> pd.DataFrame:
    """Safely loads data from an uploaded CSV or Excel file or file path.

    Args:
        file: A file path (str) or a file-like object (IO) from Streamlit's file_uploader.

    Returns:
        pd.DataFrame: The loaded DataFrame, or an empty DataFrame if loading fails.
    """
    logger.info("Attempting to load data file.")
    try:
        # Determine file type based on file name or attribute
        filename = ""
        if hasattr(file, "name"):
            filename = file.name
        elif isinstance(file, str):
            filename = file

        filename_lower = filename.lower()
        if filename_lower.endswith(".csv"):
            df = pd.read_csv(file)
            logger.info(f"Successfully loaded CSV file: {filename} with shape {df.shape}")
            return df
        elif filename_lower.endswith((".xlsx", ".xls")):
            df = pd.read_excel(file)
            logger.info(f"Successfully loaded Excel file: {filename} with shape {df.shape}")
            return df
        else:
            # Fallback attempt: try reading as CSV first, then Excel
            logger.warning("Unknown file extension. Attempting to parse as CSV.")
            try:
                df = pd.read_csv(file)
                logger.info("Successfully loaded file as CSV (fallback).")
                return df
            except Exception:
                logger.warning("Failed to parse as CSV. Attempting to parse as Excel.")
                if hasattr(file, "seek"):
                    file.seek(0)
                df = pd.read_excel(file)
                logger.info("Successfully loaded file as Excel (fallback).")
                return df

    except Exception as e:
        logger.error(f"Error loading file: {str(e)}")
        raise ValueError(f"Failed to read data file: {str(e)}")

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans the input DataFrame by performing standard pre-processing tasks:
    1. Standardizing column names (lowercase, replace spaces with underscores).
    2. Dropping duplicate rows.
    3. Filling missing numerical values with the column's mean.
    4. Fills missing categorical values with the column's mode.

    Args:
        df: The original pandas DataFrame.

    Returns:
        pd.DataFrame: A cleaned copy of the DataFrame.
    """
    logger.info("Starting automated data cleaning process.")
    
    if df.empty:
        logger.warning("Empty DataFrame passed to clean_data.")
        return df.copy()

    # Create a copy to prevent modifying the original DataFrame in place
    cleaned_df = df.copy()

    # 1. Standardize column names
    # Lowercase, strip whitespace, and replace spaces or hyphens with underscores
    cleaned_df.columns = (
        cleaned_df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", "_", regex=True)
        .str.replace("-", "_")
    )
    logger.info("Standardized column names.")

    # 2. Drop duplicate rows
    initial_rows = len(cleaned_df)
    cleaned_df = cleaned_df.drop_duplicates().reset_index(drop=True)
    duplicates_removed = initial_rows - len(cleaned_df)
    if duplicates_removed > 0:
        logger.info(f"Removed {duplicates_removed} duplicate rows.")

    # 3. & 4. Fill missing values based on column type
    for col in cleaned_df.columns:
        null_count = cleaned_df[col].isnull().sum()
        if null_count > 0:
            # Check if column is numerical
            if pd.api.types.is_numeric_dtype(cleaned_df[col]):
                mean_value = cleaned_df[col].mean()
                if pd.notna(mean_value):
                    cleaned_df[col] = cleaned_df[col].fillna(mean_value)
                    logger.info(f"Filled {null_count} missing values in numerical column '{col}' with mean: {mean_value:.4f}")
                else:
                    # If mean is NaN (all values are NaN), fill with 0
                    cleaned_df[col] = cleaned_df[col].fillna(0)
                    logger.info(f"Column '{col}' is numeric but all values are NaN. Filled with 0.")
            else:
                # Categorical / object type
                # Calculate mode safely
                mode_series = cleaned_df[col].mode()
                if not mode_series.empty:
                    mode_value = mode_series.iloc[0]
                    cleaned_df[col] = cleaned_df[col].fillna(mode_value)
                    logger.info(f"Filled {null_count} missing values in categorical column '{col}' with mode: {mode_value}")
                else:
                    # If mode is empty, fill with placeholder 'unknown'
                    cleaned_df[col] = cleaned_df[col].fillna("unknown")
                    logger.info(f"Column '{col}' is categorical but mode could not be computed. Filled with 'unknown'.")

    logger.info("Automated data cleaning process completed successfully.")
    return cleaned_df

def get_data_quality_score(df_original: pd.DataFrame, df_cleaned: pd.DataFrame) -> float:
    """Calculates a data quality score out of 100 based on the volume of missing
    values and duplicate rows that required cleaning.

    Formula:
      Total cells = num_rows_original * num_cols_original
      Score = 100 * (1 - (missing_values_count + (duplicate_rows_count * num_cols_original)) / Total cells)
      Capped at a minimum score of 0.0.

    Args:
        df_original: The raw/original pandas DataFrame.
        df_cleaned: The cleaned pandas DataFrame.

    Returns:
        float: A data quality score between 0.0 and 100.0.
    """
    logger.info("Calculating data quality score.")
    
    total_cells = df_original.size
    if total_cells == 0:
        logger.warning("Original DataFrame is empty. Returning quality score of 100.0.")
        return 100.0

    # Count missing values in the original dataset
    missing_count = int(df_original.isnull().sum().sum())

    # Count duplicate rows in the original dataset
    duplicate_count = int(df_original.duplicated().sum())

    num_cols = df_original.shape[1]
    
    # Calculate total fixed cells/issues
    # Duplicate rows account for duplicate_count * num_cols elements
    total_issues = missing_count + (duplicate_count * num_cols)

    # Calculate score
    score = 100.0 * (1.0 - (total_issues / total_cells))
    
    # Ensure score is within [0.0, 100.0]
    score = max(0.0, min(100.0, score))
    
    logger.info(f"Data Quality Score calculated: {score:.2f}/100.0 (Found {missing_count} missing cells and {duplicate_count} duplicate rows).")
    return round(score, 2)
