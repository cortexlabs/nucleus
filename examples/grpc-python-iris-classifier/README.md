## gRPC client

#### Step 1

```bash
pip install grpcio grpcio-tools
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. iris_classifier.proto
```

#### Step 2

```python
import grpc
import iris_classifier_pb2
import iris_classifier_pb2_grpc

sample = iris_classifier_pb2.Sample(
    sepal_length=5.2,
    sepal_width=3.6,
    petal_length=1.4,
    petal_width=0.3
)

grpc_endpoint = "localhost:8080"
channel = grpc.insecure_channel(grpc_endpoint)
stub = iris_classifier_pb2_grpc.HandlerStub(channel)

response = stub.Predict(sample)
print("prediction:", response.classification)
```
