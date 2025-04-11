import unittest
import mlflow
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os

class TestModelLoading(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
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

        # Load the model from model registry
        cls.model_name = "my_model"
        cls.model_version = cls.get_latest_model_version(cls.model_name)
        cls.model_uri = f"models:/{cls.model_name}/{cls.model_version}"
        cls.model = mlflow.pyfunc.load_model(cls.model_uri)

        # Load the holdout test data
        cls.holdout_data = pd.read_csv('./data/processed/test_bow.csv')

        with open('models/vectorizer.pkl', 'rb') as f:
            cls.vectorizer = pickle.load(f)


    @staticmethod
    def get_latest_model_version(model_name: str) -> int:
        """
        Fetch the latest model version by alias or fallback to highest version.
        """
        client = mlflow.MlflowClient()
        try:
            # Try to fetch version using alias
            model_version = client.get_model_version_by_alias(model_name, "latest")
            return int(model_version.version)
        except Exception as e:
            print(f"Alias 'latest' not found. Falling back to highest version. Details: {e}")
            # Fallback: Get the latest version manually
            all_versions = client.search_model_versions(f"name='{model_name}'")
            if not all_versions:
                raise ValueError(f"No versions found for model: {model_name}")
            latest = max(all_versions, key=lambda mv: int(mv.version))
            return int(latest.version)

    def test_model_loaded_properly(self):
        """Test if the model loads correctly."""
        self.assertIsNotNone(self.model, "Model should be loaded successfully")

    def test_model_signature(self):
        input_text = "hi how are you"
        input_data = self.vectorizer.transform([input_text])
        input_df = pd.DataFrame(input_data.toarray(), columns=[str(i) for i in range(input_data.shape[1])])

        # Predict using the model
        predictions = self.model.predict(input_df)

        # Verify the input shape
        self.assertEqual(input_df.shape[1], len(self.vectorizer.get_feature_names_out()))

        # verify the output shape
        self.assertEqual(len(predictions), input_df.shape[0])
        self.assertEqual(len(predictions.shape), 1)


    def test_model_performance(self):
        # Extract features and labels from holdout test data
        X_holdout = self.holdout_data.iloc[:,0:-1]
        y_holdout = self.holdout_data.iloc[:,-1]

        # Predict using the new model
        y_pred_new = self.new_model.predict(X_holdout)

        # Calculate performance metrics for the new model
        accuracy_new = accuracy_score(y_holdout, y_pred_new)
        precision_new = precision_score(y_holdout, y_pred_new)
        recall_new = recall_score(y_holdout, y_pred_new)
        f1_new = f1_score(y_holdout, y_pred_new)

        # Define expected thresholds for the performance metrics
        expected_accuracy = 0.76
        expected_precision = 0.76
        expected_recall = 0.76
        expected_f1 = 0.76

        # Assert that the new model meets the performance thresholds
        self.assertGreaterEqual(accuracy_new, expected_accuracy, f'Accuracy should be at least {expected_accuracy}')
        self.assertGreaterEqual(precision_new, expected_precision, f'Precision should be at least {expected_precision}')
        self.assertGreaterEqual(recall_new, expected_recall, f'Recall should be at least {expected_recall}')
        self.assertGreaterEqual(f1_new, expected_f1, f'F1 score should be at least {expected_f1}')


if __name__ == '__main__':
    unittest.main()
