import os
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
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
    if search_type == 'grid':
        search = GridSearchCV(model, param_grid, scoring=scoring, cv=cv)
    elif search_type == 'random':
        search = RandomizedSearchCV(model, param_grid, scoring=scoring, cv=cv, n_iter=50, random_state=42)

    search.fit(X, y)
    best_model = search.best_estimator_
    best_params = search.best_params_
    best_score = search.best_score_

    return best_model, best_params, best_score


def evaluate_models(train_df):
    """
    Evaluate multiple regression models using cross-validation and hyperparameter tuning.
    """
    features = [col for col in train_df.columns if col.startswith("corr_")]
    X = train_df[features]
    y = train_df["age"]

    models = {
        "linear_regression": LinearRegression(),
        "ridge_regression": Ridge(),
        "elastic_net": ElasticNet(),
        # "random_forest": RandomForestRegressor(),
        # "xgboost": XGBRegressor()
    }

    param_grids = {
        "linear_regression": {},
        "ridge_regression": {"alpha": [0.1, 1, 10]},
        "elastic_net": {"alpha": [0.1, 0.5, 1.0], "l1_ratio": [0.1, 0.5, 0.9]},
        # "random_forest": {"n_estimators": [50, 100, 200], "max_depth": [5, 10, 20]},
        # "xgboost": {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 0.2], "max_depth": [3, 6, 9]}
    }

    best_models = {}
    results = {}

    # Create trained_models directory if it doesn't exist
    os.makedirs("trained_models", exist_ok=True)

    for name, model in models.items():
        print(f"Evaluating {name}...")

        best_model, best_params, best_score = tune_and_evaluate_model(
            model, param_grids[name], X, y, search_type='grid', cv=2
        )

        best_models[name] = best_model
        results[name] = {
            "best_params": best_params,
            "best_score": best_score
        }

        print(f"{name} - Best Parameters: {best_params}, Best CV Score: {best_score:.4f}")

        # Save the best model
        joblib.dump(best_model, f"trained_models/{name}_model.joblib")

        # Save the model's parameters
        with open(f"trained_models/{name}_params.json", "w") as f:
            json.dump(best_params, f)

    return best_models, results


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


def load_and_predict(test_df, model_name):
    """
    Load a pre-trained model and make predictions
    """
    # Load the trained model
    try:
        model = joblib.load(f"trained_models/{model_name}_model.joblib")

        # Load model parameters (optional, for reference)
        with open(f"trained_models/{model_name}_params.json", "r") as f:
            model_params = json.load(f)
        print(f"Loaded {model_name} model with parameters: {model_params}")

        # Prepare test features
        features = [col for col in test_df.columns if col.startswith("corr_")]
        X_test = test_df[features]

        # Make predictions
        predictions = model.predict(X_test)

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

    # Option 1: Train and save models
    print("Training and saving models...")
    best_models, cv_results = evaluate_models(train_df)

    # Train and predict with all models
    for name, model in best_models.items():
        train_and_predict(train_df, test_df, name, model)

    # # Option 2: Load and predict with pre-trained models
    # print("\nLoading pre-trained models and making predictions...")
    # model_names = ["linear_regression", "ridge_regression", "elastic_net", "random_forest", "xgboost"]
    # for model_name in model_names:
    #     load_and_predict(test_df, model_name)


if __name__ == "__main__":
    main()