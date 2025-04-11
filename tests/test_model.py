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
        Fetch the latest version number of a registered MLflow model.
        """
        client = mlflow.MlflowClient()
        all_versions = client.get_model_version_by_alias(model_name, stages=["None"])
        
        if not all_versions:
            raise ValueError(f"No versions found for model: {model_name}")
        
        latest_version = max(all_versions, key=lambda mv: int(mv.version))
        return int(latest_version.version)
    
    def test_model_loaded_properly(self):
        """Test if the model loads correctly."""
        self.assertIsNotNone(self.model, "Model should be loaded successfully")

if __name__ == '__main__':
    unittest.main()