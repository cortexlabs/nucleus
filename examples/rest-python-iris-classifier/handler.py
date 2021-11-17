import os
import pickle

labels = ["setosa", "versicolor", "virginica"]

class Handler:
    def __init__(self, config, model_client):
        self.client = model_client

    def load_model(self, model_path):
        return pickle.load(open(os.path.join(model_path, "model.pkl"), "rb"))

    def handle_post(self, payload, query_params):
        model_version = query_params.get("version", "latest")

        model = self.client.get_model(model_version=model_version)
        model_input = [
            payload["sepal_length"],
            payload["sepal_width"],
            payload["petal_length"],
            payload["petal_width"],
        ]
        label_index = model.predict([model_input]).item()

        return {"prediction": labels[label_index], "model": {"version": model_version}}
