from flask import Flask, render_template, request
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
import pickle
import os
import sys

# Add the parent directory to sys.path so Python can locate preprocessing_utility
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from preprocessing_utility import normalize_text

def get_latest_model_version(model_name: str) -> int:
    """
    Fetch the latest version number of a registered MLflow model.

    Args:
        model_name (str): Name of the registered model.
        tracking_uri (str, optional): MLflow tracking URI. Defaults to the local setup.

    Returns:
        int: Latest version number of the registered model.

    Raises:
        ValueError: If the model does not exist or has no versions.
    """

    client = MlflowClient()

    all_versions = client.search_model_versions(f"name='{model_name}'")

    if not all_versions:
        raise ValueError(f"No versions found for model: {model_name}")

    latest_version = max(all_versions, key=lambda mv: int(mv.version))
    return int(latest_version.version)


# Set up DagsHub credentials for MLflow tracking
dagshub_token = os.getenv("DAGSHUB_PAT")
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "Prayesh13"
repo_name = "mlops-mini-project"

# Set up MLflow tracking URI
mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

model_name = "my_model"   # name of the model is based on that company will register the model
model_version = get_latest_model_version(model_name) # version of the model
model_uri = f"models:/{model_name}/{model_version}"
model = mlflow.pyfunc.load_model(model_uri)  # load the model from model registry

vectorizer = pickle.load(open('models/vectorizer.pkl', 'rb'))  # load the vectorizer

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html', result=None)

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']

    # clean the input text
    text = normalize_text(text)

    # now apply bow on input text
    features = vectorizer.transform([text])  # transform the text using the vectorizer

    # Convert sparse matrix to DataFrame
    features_df = pd.DataFrame.sparse.from_spmatrix(features)
    features_df = pd.DataFrame(features.toarray(), columns=[str(i) for i in range(features.shape[1])])
    
    # predict using the model
    result = model.predict(features_df)  # predict using the model

    # show  the output to the user
    return render_template("index.html",result = result[0])

if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True, port=8000)