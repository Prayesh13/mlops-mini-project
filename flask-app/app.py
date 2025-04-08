from flask import Flask, render_template, request
import mlflow
from mlflow.tracking import MlflowClient
from preprocessing_utility import normalize_text
import pickle

app = Flask(__name__)


def get_latest_model_version(model_name: str, tracking_uri: str = None) -> int:
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
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    client = MlflowClient()

    all_versions = client.search_model_versions(f"name='{model_name}'")

    if not all_versions:
        raise ValueError(f"No versions found for model: {model_name}")

    latest_version = max(all_versions, key=lambda mv: int(mv.version))
    return int(latest_version.version)

# load the model from model registry
tracking_uri = "https://dagshub.com/Prayesh13/mlops-mini-project.mlflow"
model_name = "my_model"   # name of the model is based on that company will register the model
model_version = get_latest_model_version(model_name, tracking_uri) # version of the model
model_uri = f"models:/{model_name}/{model_version}"
model = mlflow.pyfunc.load_model(model_uri)  # load the model from model registry

vectorizer = pickle.load(open('models/vectorizer.pkl', 'rb'))  # load the vectorizer


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
    
    # predict using the model
    result = model.predict(features)  # predict using the model

    # show  the output to the user
    return render_template("index.html",result = result[0])

app.run(debug=True, port=8000)