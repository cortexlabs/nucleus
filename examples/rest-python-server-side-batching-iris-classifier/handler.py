import re
import torch
import boto3
from botocore import UNSIGNED
from botocore.client import Config
from model import IrisNet

labels = ["setosa", "versicolor", "virginica"]


class Handler:
    def __init__(self, config):
        # extract s3 model path
        bucket, key = re.match("s3://(.+?)/(.+)", config["model"]).groups()

        # download the model
        s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
        s3.download_file(bucket, key, "/tmp/model.pth")

        # initialize the model
        model = IrisNet()
        model.load_state_dict(torch.load("/tmp/model.pth"))
        model.eval()

        self.model = model

    def handle_post(self, payload):
        responses = []

        # note: this is not the most efficient way, it's just to test server-side batching
        for sample in payload:
            # Convert the request to a tensor and pass it into the model
            input_tensor = torch.FloatTensor(
                [
                    [
                        sample["sepal_length"],
                        sample["sepal_width"],
                        sample["petal_length"],
                        sample["petal_width"],
                    ]
                ]
            )

            # Run the prediction
            output = self.model(input_tensor)

            # Translate the model output to the corresponding label string
            responses.append(labels[torch.argmax(output[0])])

        return responses
