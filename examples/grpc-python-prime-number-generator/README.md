## Prime number generator

#### Step 1

```bash
pip install grpcio grpcio-tools
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. generator.proto
```

#### Step 2

```python
import grpc
import generator_pb2
import generator_pb2_grpc

grpc_endpoint = "localhost:8080"
channel = grpc.insecure_channel(grpc_endpoint)
stub = generator_pb2_grpc.GeneratorStub(channel)
for r in stub.Predict(generator_pb2.Input(prime_numbers_to_generate=5)):
    print(r)
```
