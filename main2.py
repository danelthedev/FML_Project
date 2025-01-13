import os
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model._stochastic_gradient import DEFAULT_EPSILON
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import StratifiedKFold
import json


def extract_upper_triangular(file_path):
    """
    Extract the upper triangular part of the matrix from a .tsv file and return it as a flattened vector.
    """
    matrix = pd.read_csv(file_path, sep="\t", header=None)
    upper_triangular = np.triu(matrix.values, k=1)
    correlation_vector = upper_triangular[np.triu_indices_from(upper_triangular, k=1)]
    return correlation_vector


def get_participant_id_from_filename(filename):
    """
    Extract the participant_id from the filename.
    """
    return filename.split("_")[0].replace("sub-", "")


def create_dataframes(metadata_file, tsv_folder):
    """
    Create a dataframe of correlation vectors by extracting the upper triangular portion from .tsv files in a folder.
    """
    metadata = pd.read_csv(metadata_file)
    correlation_data = []

    for file_name in os.listdir(tsv_folder):
        if file_name.endswith(".tsv"):
            participant_id = get_participant_id_from_filename(file_name)
            correlation_vector = extract_upper_triangular(os.path.join(tsv_folder, file_name))
            correlation_data.append([participant_id] + list(correlation_vector))

    correlation_columns = [f"corr_{i}" for i in range(len(correlation_data[0]) - 1)]
    correlation_df = pd.DataFrame(correlation_data, columns=["participant_id"] + correlation_columns)
    final_df = pd.merge(metadata, correlation_df, on="participant_id", how="inner")
    return final_df


def tune_and_evaluate_model(model, param_grid, X, y, scoring='neg_mean_squared_error', cv=5, search_type='grid'):
    """
    Perform grid search or randomized search with cross-validation.
    """
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if search_type == 'grid':
        search = GridSearchCV(model, param_grid, scoring=scoring, cv=cv, n_jobs=-1)
    elif search_type == 'random':
        search = RandomizedSearchCV(model, param_grid, scoring=scoring, cv=cv, n_iter=100,
                                    random_state=42, n_jobs=-1)

    search.fit(X_scaled, y)
    best_model = search.best_estimator_
    best_params = search.best_params_
    best_score = search.best_score_

    return best_model, best_params, best_score, scaler


def evaluate_models(train_df):
    """
    Evaluate SGDRegressor using cross-validation and hyperparameter tuning.
    """
    features = [col for col in train_df.columns if col.startswith("corr_")]
    X = train_df[features]
    y = train_df["age"]

    # Initialize SGDRegressor with reasonable defaults
    models = {
        "sgd_regression": SGDRegressor(max_iter=5000, random_state=42)
    }

    # Comprehensive parameter grid for SGDRegressor
    param_grids = {
        "sgd_regression": {
            "loss": ["huber"],
            "penalty": ["elasticnet"],
            "alpha": [0.00065, 0.0006, 0.000625],
            "learning_rate": ["optimal"],
            "eta0": [0.045, 0.05, 0.06, 0.7],
            "l1_ratio": [0.275, 0.3, 0.325, 0.35],
            "epsilon": [0.05, 0.055, 0.045, 0.4, 0.6],  # for huber loss
            "tol": [0.0001, 0.00015, 0.0002, 0.00005]
        }
    }

    best_models = {}
    results = {}
    scalers = {}

    # Create trained_models directory if it doesn't exist
    os.makedirs("trained_models", exist_ok=True)

    for name, model in models.items():
        print(f"Evaluating {name}...")

        best_model, best_params, best_score, scaler = tune_and_evaluate_model(
            model, param_grids[name], X, y, search_type='random', cv=5
        )

        best_models[name] = best_model
        scalers[name] = scaler
        results[name] = {
            "best_params": best_params,
            "best_score": best_score
        }

        print(f"{name} - Best Parameters: {best_params}")
        print(f"{name} - Best CV Score: {best_score:.4f}")

        # Save the best model and scaler
        joblib.dump(best_model, f"trained_models/{name}_model.joblib")
        joblib.dump(scaler, f"trained_models/{name}_scaler.joblib")

        # Save the model's parameters
        with open(f"trained_models/{name}_params.json", "w") as f:
            json.dump(best_params, f)

    return best_models, results, scalers


def train_and_predict(train_df, test_df, model_name, model, scaler):
    """
    Train a model and make predictions
    """
    features = [col for col in train_df.columns if col.startswith("corr_")]
    X_train = train_df[features]
    y_train = train_df["age"]
    X_test = test_df[features]

    # Scale the features
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    model.fit(X_train_scaled, y_train)
    predictions = model.predict(X_test_scaled)

    # Save predictions
    predictions_df = pd.DataFrame(
        {"participant_id": test_df["participant_id"], "age": predictions}
    )
    predictions_df.to_csv(f"{model_name}_predictions.csv", index=False)
    print(f"Predictions saved to '{model_name}_predictions.csv'")


def load_and_predict(test_df, model_name):
    """
    Load a pre-trained model and make predictions
    """
    try:
        # Load the trained model and scaler
        model = joblib.load(f"trained_models/{model_name}_model.joblib")
        scaler = joblib.load(f"trained_models/{model_name}_scaler.joblib")

        # Load model parameters
        with open(f"trained_models/{model_name}_params.json", "r") as f:
            model_params = json.load(f)
        print(f"Loaded {model_name} model with parameters: {model_params}")

        # Prepare and scale test features
        features = [col for col in test_df.columns if col.startswith("corr_")]
        X_test = test_df[features]
        X_test_scaled = scaler.transform(X_test)

        # Make predictions
        predictions = model.predict(X_test_scaled)

        # Save predictions
        predictions_df = pd.DataFrame(
            {"participant_id": test_df["participant_id"], "age": predictions}
        )
        predictions_df.to_csv(f"{model_name}_predictions.csv", index=False)
        print(f"Predictions saved to '{model_name}_predictions.csv'")

        return predictions

    except FileNotFoundError:
        print(f"Error: No trained model found for {model_name}. Please train the model first.")
        return None


def main():
    # Define paths for metadata files and tsv folders
    train_metadata_file = "metadata/training_metadata.csv"
    test_metadata_file = "metadata/test_metadata.csv"
    train_tsv_folder = "train_tsv/train_tsv"
    test_tsv_folder = "test_tsv/test_tsv"

    # Load the dataframes
    train_df = create_dataframes(train_metadata_file, train_tsv_folder)
    test_df = create_dataframes(test_metadata_file, test_tsv_folder)

    # Train and save models
    print("Training and saving models...")
    best_models, cv_results, scalers = evaluate_models(train_df)

    # Train and predict with all models
    for name, model in best_models.items():
        train_and_predict(train_df, test_df, name, model, scalers[name])


if __name__ == "__main__":
    main()