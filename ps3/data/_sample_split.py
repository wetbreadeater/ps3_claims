#import hashlib
import pandas as pd
import numpy as np

# TODO: Write a function which creates a sample split based in some id_column and training_frac.
# Optional: If the dtype of id_column is a string, we can use hashlib to get an integer representation.
def create_sample_split(df, id_columns, training_frac=0.8):
    """Create sample split based on ID column.

    Parameters
    ----------
    df : pd.DataFrame
        Training data
    id_column : str
        Name of ID column
    training_frac : float, optional
        Fraction to use for training, by default 0.9

    Returns
    -------
    pd.DataFrame
        Training data with sample column containing train/test split based on IDs.
    """
    # Check if id_column is a string
    if isinstance(id_columns, str):
        id_columns = [id_columns]

    # Randomise the data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Assign a random number to each row
    if id_columns:
        # Create a unique identifier for each row
        df["unique_id"] = df[id_columns].astype(str).agg("".join, axis=1)

        # Map each group to a unqiue identifier
        group_map = {
            group: np.random.uniform(0, 1) for group in df["unique_id"].unique()
        }
        df["random_"] = df["unique_id"].map(group_map)
    else:
        # Create unique identifier for each row if no split columns are provided
        df["random_"] = np.random.uniform(0, 1, size=len(df))
    # Create sample column based on training_frac
    df["sample"] = np.where(df["random_"] <= training_frac, "train", "test")
     # Drop the temporary columns
    df = df.drop(columns=["random_", "unique_id"], errors="ignore")

    return df
