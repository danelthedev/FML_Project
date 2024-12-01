import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import json


def extract_upper_triangular(file_path):
    """
    Extract the upper triangular part of the matrix from a .tsv file and return it as a flattened vector.
    """
    # Read the tsv file as a pandas DataFrame
    matrix = pd.read_csv(file_path, sep="\t", header=None)

    # Get the upper triangular portion of the matrix
    upper_triangular = np.triu(matrix.values, k=1)  # k=1 skips the diagonal

    # Flatten the upper triangular portion to get the correlation vector
    correlation_vector = upper_triangular[np.triu_indices_from(upper_triangular, k=1)]

    return correlation_vector


def get_participant_id_from_filename(filename):
    """
    Extract the participant_id from the filename, which follows the format 'sub-{participantId}_ses-restOfFileName.tsv'.
    """
    return filename.split("_")[0].replace("sub-", "")


def create_dataframes(metadata_file, tsv_folder):
    """
    Create a dataframe of correlation vectors by extracting the upper triangular portion from .tsv files in a folder
    and merging it with the metadata.
    """
    # Load the metadata
    metadata = pd.read_csv(metadata_file)

    # Prepare a list to store the correlation vectors and the corresponding participant IDs
    correlation_data = []

    # Loop through all .tsv files in the given folder
    for file_name in os.listdir(tsv_folder):
        if file_name.endswith(".tsv"):
            # Get the participant ID from the file name
            participant_id = get_participant_id_from_filename(file_name)

            # Get the correlation vector from the .tsv file
            correlation_vector = extract_upper_triangular(
                os.path.join(tsv_folder, file_name)
            )

            # Append the correlation vector along with the participant ID
            correlation_data.append([participant_id] + list(correlation_vector))

    # Create a DataFrame from the correlation data
    correlation_columns = [
        f"corr_{i}" for i in range(len(correlation_data[0]) - 1)
    ]  # Column names for correlations
    correlation_df = pd.DataFrame(
        correlation_data, columns=["participant_id"] + correlation_columns
    )

    # Merge the correlation DataFrame with the metadata based on participant_id
    final_df = pd.merge(metadata, correlation_df, on="participant_id", how="inner")

    return final_df


# Paths to metadata files and tsv folders
train_metadata_file = "metadata/training_metadata.csv"
test_metadata_file = "metadata/test_metadata.csv"
train_tsv_folder = "train_tsv/train_tsv"
test_tsv_folder = "test_tsv/test_tsv"

# Create dataframes for training and test data
train_df = create_dataframes(train_metadata_file, train_tsv_folder)
test_df = create_dataframes(test_metadata_file, test_tsv_folder)

# Show the resulting dataframe
print(train_df)
print(test_df)


def evaluate_models(train_df):
    """
    Evaluate multiple regression models using cross-validation
    """
    # Prepare features and target
    features = [col for col in train_df.columns if col.startswith("corr_")]
    X = train_df[features]
    y = train_df["age"]

    # Initialize models
    models = {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "xgboost": XGBRegressor(random_state=42),
    }

    # Evaluate each model
    results = {}
    for name, model in models.items():
        # Perform 5-fold cross-validation
        cv_scores = cross_val_score(model, X, y, cv=5, scoring="neg_mean_squared_error")
        rmse_scores = np.sqrt(-cv_scores)

        results[name] = {
            "mean_rmse": float(rmse_scores.mean()),
            "std_rmse": float(rmse_scores.std()),
            "cv_rmse_scores": rmse_scores.tolist(),
        }

        # Save individual model results
        with open(f"{name}_cv_results.json", "w") as f:
            json.dump(results[name], f, indent=4)

        print(
            f"{name.upper()} - Mean RMSE: {rmse_scores.mean():.2f} (±{rmse_scores.std():.2f})"
        )

    return models, results


def train_and_predict(train_df, test_df, model_name, model):
    """
    Train a model and make predictions
    """
    features = [col for col in train_df.columns if col.startswith("corr_")]
    X_train = train_df[features]
    y_train = train_df["age"]
    X_test = test_df[features]

    # Train model
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # Save predictions
    predictions_df = pd.DataFrame(
        {"participant_id": test_df["participant_id"], "age": predictions}
    )
    predictions_df.to_csv(f"{model_name}_predictions.csv", index=False)
    print(f"Predictions saved to '{model_name}_predictions.csv'")


# Evaluate models
models, cv_results = evaluate_models(train_df)

# Train and predict with each model
for name, model in models.items():
    train_and_predict(train_df, test_df, name, model)
