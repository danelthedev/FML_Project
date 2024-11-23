import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression



def extract_upper_triangular(file_path):
    """
    Extract the upper triangular part of the matrix from a .tsv file and return it as a flattened vector.
    """
    # Read the tsv file as a pandas DataFrame
    matrix = pd.read_csv(file_path, sep='\t', header=None)

    # Get the upper triangular portion of the matrix
    upper_triangular = np.triu(matrix.values, k=1)  # k=1 skips the diagonal

    # Flatten the upper triangular portion to get the correlation vector
    correlation_vector = upper_triangular[np.triu_indices_from(upper_triangular, k=1)]

    return correlation_vector


def get_participant_id_from_filename(filename):
    """
    Extract the participant_id from the filename, which follows the format 'sub-{participantId}_ses-restOfFileName.tsv'.
    """
    return filename.split('_')[0].replace('sub-', '')


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
        if file_name.endswith('.tsv'):
            # Get the participant ID from the file name
            participant_id = get_participant_id_from_filename(file_name)

            # Get the correlation vector from the .tsv file
            correlation_vector = extract_upper_triangular(os.path.join(tsv_folder, file_name))

            # Append the correlation vector along with the participant ID
            correlation_data.append([participant_id] + list(correlation_vector))

    # Create a DataFrame from the correlation data
    correlation_columns = [f"corr_{i}" for i in range(len(correlation_data[0]) - 1)]  # Column names for correlations
    correlation_df = pd.DataFrame(correlation_data, columns=["participant_id"] + correlation_columns)

    # Merge the correlation DataFrame with the metadata based on participant_id
    final_df = pd.merge(metadata, correlation_df, on="participant_id", how="inner")

    return final_df


# Paths to metadata files and tsv folders
train_metadata_file = 'metadata/training_metadata.csv'
test_metadata_file = 'metadata/test_metadata.csv'
train_tsv_folder = 'train_tsv/train_tsv'
test_tsv_folder = 'test_tsv/test_tsv'

# Create dataframes for training and test data
train_df = create_dataframes(train_metadata_file, train_tsv_folder)
test_df = create_dataframes(test_metadata_file, test_tsv_folder)

# Show the resulting dataframe
print(train_df)
print(test_df)

def predict_age(train_df, test_df):
    # Assuming 'age' column is present in the metadata
    # Separate the features (correlation columns) and target (age) for training
    features = [col for col in train_df.columns if col.startswith('corr_')]  # Select correlation columns
    X_train = train_df[features]  # Training features (correlation vectors)
    y_train = train_df['age']  # Target variable (age)

    # Initialize and train a linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Now, let's predict the age for the test dataset
    X_test = test_df[features]  # Test features (correlation vectors)
    predictions = model.predict(X_test)  # Predict age for the test set

    # Store the predictions with participant_id
    predictions_df = pd.DataFrame({
        'participant_id': test_df['participant_id'],  # Extract participant IDs
        'age': predictions  # Predicted ages
    })

    # Save the predictions to a CSV file
    predictions_df.to_csv('predictions.csv', index=False)
    print("Predictions saved to 'predictions.csv'")


# Assuming train_df and test_df are already created and contain 'age' in metadata
# Call the function to predict age and store the predictions
predict_age(train_df, test_df)
