import unittest
import mlflow
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

if __name__ == '__main__':
    unittest.main()
