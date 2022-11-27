# Nucleus model server

Nucleus is a model server for TensorFlow and generic Python models. It is compatible with [Cortex](https://github.com/cortexlabs/cortex) clusters, Kubernetes clusters, and any other container-based deployment platforms. Nucleus can also be run locally via docker compose.

Some of Nucleus's features include:

* Generic Python models (PyTorch, ONNX, Sklearn, MLFlow, Numpy, Pandas, Caffe, etc)
* TensorFlow models
* CPU and GPU support
* Serve models directly from S3 paths
* Multiprocessing and multithreadding
* Multi-model endpoints
* Dynamic server-side request batching
* Automatic model reloading when new model versions are uploaded to S3
* Model caching based on LRU policy (disk and memory)
* HTTP and gRPC support

# Table of contents

* [Getting started](#getting-started)
  * [Install Nucleus](#install-nucleus)
  * [Example usage](#example-usage)
* [Configuration](#configuration)
  * [Model server configuration schema (nucleus.yaml)](#model-server-configuration-schema)
  * [Project files](#project-files)
    * [PyPI packages](#pypi-packages)
    * [Conda packages](#conda-packages)
    * [GitHub packages](#github-packages)
    * [setup.py](#setuppy)
    * [System packages](#system-packages)
    * [Example project structure](#example-project-structure)
  * [Pod integration](#pod-integration)
    * [Cortex cluster](#cortex-cluster)
    * [Non-Cortex cluster](#non-cortex-cluster)
  * [Multi-model](#multi-model)
  * [Multi-model caching](#multi-model-caching)
  * [Server-side batching](#server-side-batching)
* [Handler implementation](#handler-implementation)
  * [HTTP](#http)
  * [gRPC](#grpc)
  * [Nucleus parallelism](#nucleus-parallelism)
  * [Models](#models)
    * [Python Handler](#python-handler)
    * [Tensorflow Handler](#tensorflow-handler)
  * [Structured logging](#structured-logging)
* Examples
  * [REST Python (iris classifier)](examples/rest-python-iris-classifier)
  * [REST Python model caching (translator model)](examples/rest-python-model-caching-translator)
  * [REST Python server-side batching (iris classifier)](examples/rest-python-server-side-batching-iris-classifier)
  * [REST TensorFlow Serving multi-model (generic classifiers)](examples/rest-tfs-multi-model-classifier)
  * [REST TensorFlow Serving multi-model caching (generic classifiers)](examples/rest-tfs-multi-model-caching-classifier)
  * [REST TensorFlow Serving server-side batching (inception classifier)](examples/rest-tfs-server-side-batching-inception-classifier)
  * [gRPC Python (iris classifier)](examples/grpc-python-iris-classifier)
  * [gRPC Python (prime number generator)](examples/grpc-python-prime-generator)

# Getting started

## Install Nucleus

<!-- UPDATE THIS VERSION ON EACH RELEASE (it's better than using "master") -->

```bash
pip install git+https://github.com/cortexlabs/nucleus.git@0.2.2
```

## Example usage

Generate a model server Dockerfile based on a Nucleus configuration file:

```bash
nucleus generate examples/rest-python-iris-classifier/nucleus.yaml
```

Build the Docker image:

```bash
docker build -f nucleus.Dockerfile -t iris-classifier .
```

Run the Nucleus model server:

```bash
docker run -it --rm -p 8080:8080 iris-classifier
```

Make a request to the model server:

```bash
curl localhost:8080/ -X POST -H "Content-type: application/json" -d @examples/rest-python-iris-classifier/sample.json
```

# Configuration

To build a Nucleus model server, you will need a directory which contains your source code as well as a model server configuration file (e.g. `nucleus.yaml`).

## Model server configuration schema

```yaml
type: <string>  # python or tensorflow (required)

# versions
py_version: <string>  # python version (default: 3.6.9)
tfs_version: <string>  # tensorflow version (only applicable for tensorflow model type) (default: 2.3.0)

# gpu
gpu: <bool>  # whether your model server will use a GPU (default: false)
gpu_version:  # gpu version (only applicable for python model type, required if gpu is used)
  cuda: <string>  # cuda version (tested with 10.0, 10.1, 10.2, 11.0, 11.1) (required)
  cudnn: <string>  # cudnn version (tested with 7 and 8) (required)
  development_image: <bool>  # whether to use the dev cuda image or not (default: false)

# dependencies
dependencies:
  pip: <string>  # relative path to requirements.txt (default: requirements.txt)
  conda: <string>  # relative path to conda-packages.txt (default: conda-packages.txt)
  shell: <string>  # relative path to a shell script for system package installation (default: dependencies.sh)

serve_port: <int>  # the port that will be responding to requests (default: 8080)
env:  # environment variables to be exported to the model server (optional)
  # <string: string>  # dictionary of environment variables
config:  # dictionary config to be passed to the model server's constructor (optional)
  # <string: value>  # arbitrary dictionary passed to the constructor of the Handler class
python_path: <string>  # path to the root of your Python folder that will be appended to PYTHONPATH (default: folder containing the model server config) (required)
path: <string>  # path to the Python handler script containing the Handler class relative to the <python_path> path (required)
protobuf_path: <string>  # path to a protobuf file relative to your project dir (required if using gRPC)
models: # use this to serve one or more models with live reloading for the tensorflow type (required for tensorflow type, not supported for python type)
  path: <string>  # S3 path to an exported model directory (e.g. s3://my-bucket/exported_model/) (either this, 'dir', or 'paths' must be provided if 'models' is specified)
  paths:  # list of S3 paths to exported model directories (either this, 'dir', or 'path' must be provided if 'models' is specified)
    - name: <string>  # unique name for the model (e.g. text-generator) (required)
      path: <string>  # S3 path to an exported model directory (e.g. s3://my-bucket/exported_model/) (required)
  dir: <string>  # S3 path to a directory containing multiple models (e.g. s3://my-bucket/models/) (either this, 'path', or 'paths' must be provided if 'models' is specified)
  cache_size: <int>  # the number models to keep in memory (optional; all models are kept in memory by default)
  disk_cache_size: <int>  # the number of models to keep on disk (optional; all models are kept on disk by default)
multi_model_reloading:  # use this to serve one or more models with live reloading for the python type (optional for python type, not supported for tensorflow type)
  - ...  # same schema as for "models"

# concurrency
processes: <int>  # the number of parallel serving processes to run on each Nucleus instance (default: 1)
threads_per_process: <int>  # the number of threads per process (default: 1)
max_concurrency: <int>  # max concurrency (default: 0, i.e. disabled)

# server side batching
server_side_batching:  # (optional)
  max_batch_size: <int>  # the maximum number of requests to aggregate before running inference
  batch_interval_seconds: <float>  # the maximum amount of time in seconds to spend waiting for additional requests before running inference on the batch of requests

# misc
tfs_container_dns: <string>  # the address of the tensorflow serving container for when the tensorflow type is used; only necessary when developing locally using containers/docker-compose; for K8s pods, this is not necessary (default: "localhost")
log_level: <string>  # log level that can be "debug", "info", "warning" or "error" (default: "info")
use_local_cortex_libs: <bool>  # use the local cortex libs, useful when developing the nucleus model server (default: false)
```

## Project files

Nucleus makes all files in the project directory (i.e. the directory which contains the model server configuration file) available for use in your Handler class implementation. At runtime, the working directory will be set to the root of your project directory.

### PyPI packages

Nucleus looks for a `requirements.txt` file in the top level of your project directory, and will use `pip` to install the listed packages within your container.

For example:

```text
./my-classifier/
├── nucleus.yaml
├── handler.py
├── ...
└── requirements.txt
```

#### Private PyPI packages

To install packages from a private PyPI index, create a `pip.conf` inside the same directory as `requirements.txt`, and add the following contents:

```text
[global]
extra-index-url = https://<username>:<password>@<my-private-index>.com/pip
```

In same directory, create a [`dependencies.sh` script](system-packages.md) and add the following:

```bash
cp pip.conf /etc/pip.conf
```

You may now add packages to `requirements.txt` which are found in the private index.

### Conda packages

Nucleus supports installing Conda packages. We recommend only using Conda when your required packages are not available in PyPI. Nucleus looks for a `conda-packages.txt` file in the top level Nucleus project directory:

```text
./my-classifier/
├── nucleus.yaml
├── handler.py
├── ...
└── conda-packages.txt
```

The `conda-packages.txt` file follows the format of `conda list --export`. Each line of `conda-packages.txt` should follow this pattern: `[channel::]package[=version[=buildid]]`.

Here's an example of `conda-packages.txt`:

```text
conda-forge::rdkit
conda-forge::pygpu
```

In situations where both `requirements.txt` and `conda-packages.txt` are provided, Nucleus installs Conda packages in `conda-packages.txt` followed by PyPI packages in `requirements.txt`. Conda and Pip package managers install packages and dependencies independently. You may run into situations where Conda and pip package managers install different versions of the same package because they install and resolve dependencies independently from one another. To resolve package version conflicts, it may be in your best interest to specify their exact versions in `conda-packages.txt`.

### GitHub packages

You can also install public/private packages from git registries (such as GitHub) by adding them to `requirements.txt`. Here's an example for GitHub:

```text
# requirements.txt

# public access
git+https://github.com/<username>/<repo name>.git@<tag or branch name>#egg=<package name>

# private access
git+https://<personal access token>@github.com/<username>/<repo name>.git@<tag or branch name>#egg=<package name>
```

On GitHub, you can generate a personal access token by following [these steps](https://help.github.com/en/github/authenticating-to-github/creating-a-personal-access-token-for-the-command-line).

### setup.py

Python packages can also be installed by providing a `setup.py` that describes your project's modules. Here's an example directory structure:

```text
./my-classifier/
├── nucleus.yaml
├── handler.py
├── ...
├── mypkg
│   └── __init__.py
├── requirements.txt
└── setup.py
```

In this case, `requirements.txt` will have this form:

```text
# requirements.txt

.
```

### System packages

Nucleus looks for a file named `dependencies.sh` in the top level project directory. For example:

```text
./my-classifier/
├── nucleus.yaml
├── handler.py
├── ...
└── dependencies.sh
```

`dependencies.sh` is executed with `bash` shell at build time (before installing Python packages in `requirements.txt` or `conda-packages.txt`). Typical use cases include installing required system packages to be used in your Handler, building Python packages from source, etc.

Here is an example `dependencies.sh`, which installs the `tree` utility:

```bash
apt-get update && apt-get install -y tree
```

The `tree` utility can now be called inside your `handler.py`:

```python
# handler.py

import subprocess

class Handler:
    def __init__(self, config):
        subprocess.run(["tree"])
    ...
```

### Example project structure

Here is a sample project directory

```text
./my-classifier/
├── nucleus.yaml
├── handler.py
├── my-data.json
├── ...
├── .env
├── dependencies.sh
├── conda-packages.txt
└── requirements.txt
```

The `dependencies.sh` file will be executed and the Python packages specified in `requirements.txt` and `conda-packages.txt` will be installed in your docker image at build time.

At run time, you can access the environment variables specified in `.env`, and you can access `my-data.json` in your handler class like this:

```python
# handler.py

import json

class Handler:
    def __init__(self, config):
        with open('my-data.json', 'r') as data_file:
            data = json.load(data_file)
        self.data = data
```

## Pod integration

### Cortex cluster

Here is an example of a Cortex api configuration for deploying to a [Cortex cluster](https://github.com/cortexlabs/cortex):

```yaml
- name: iris-classifier
  kind: RealtimeAPI
  pod:
    port: 8080
    max_concurrency: 1
    containers:
    - name: nucleus
      image: quay.io/my-org/iris-classifier-nucleus:latest
      readiness_probe:
        exec:
          command: ["ls", "/run/workspace/api_readiness.txt"]
      compute:
        cpu: 200m
        mem: 256Mi
```

When deploying a Nucleus model server to a [Cortex cluster](https://github.com/cortexlabs/cortex), there are a few things to keep in mind:

* It is highly recommended to configure a readiness probe (if omitted, requests may be routed to your server before it's ready). This can be done by checking when `/run/workspace/api_readiness.txt` is present on the filesystem. See above for an example.
* When using the tensorflow model type, the TensorFlow Serving container will be listening on port 9000. Make sure no other service uses this port within the pod.
* The metadata of all loaded models (when the `models` or the `multi_model_reloading` fields are specified) is made available at `/info`.

### Non-Cortex cluster

When deploying a Nucleus model server with the tensorflow type on a generic Kubernetes pod (not within Cortex), there are some additional things to keep in mind:

* A shared volume (at `/mnt`) must exist between the handler container and the TensorFlow Serving container.
* The host of the TensorFlow Serving container has to be specified in the model server configuration (`nucleus.yaml`) so that the handler container can connect to it.

## Multi-model

### Python Handler

#### Specifying models in Nucleus configuration

##### `nucleus.yaml`

The directory `s3://cortex-examples/sklearn/mpg-estimator/linreg/` contains 4 different versions of the model.

```yaml
# nucleus.yaml

type: python
path: handler.py
models:
  path: s3://cortex-examples/sklearn/mpg-estimator/linreg/
```

##### `handler.py`

```python
import mlflow.sklearn


class Handler:
    def __init__(self, config, python_client):
        self.client = python_client

    def load_model(self, model_path):
        return mlflow.sklearn.load_model(model_path)

    def handle_post(self, payload, query_params):
        model_version = query_params.get("version")

        # model_input = ...

        model = self.client.get_model(model_version=model_version)
        result = model.predict(model_input)

        return {"prediction": result, "model": {"version": model_version}}
```

#### Without specifying models in Nucleus configuration

##### `nucleus.yaml`

```yaml
# nucleus.yaml

type: python
path: handler.py
...
```

##### `handler.py`

```python
class Handler:
    def __init__(self, config):
        self.analyzer = initialize_model("sentiment-analysis")
        self.summarizer = initialize_model("summarization")

    def handle_post(self, query_params, payload):
        model_name = query_params.get("model")
        model_input = payload["text"]

        # ...

        if model_name == "analyzer":
            results = self.analyzer(model_input)
            predicted_label = postprocess(results)
            return {"label": predicted_label}
        elif model_name == "summarizer":
            results = self.summarizer(model_input)
            predicted_label = postprocess(results)
            return {"label": predicted_label}
        else:
            return JSONResponse({"error": f"unknown model: {model_name}"}, status_code=400)
```

### TensorFlow Handler

#### `nucleus.yaml`

```yaml
# nucleus.yaml

type: tensorflow
path: handler.py
models:
  paths:
    - name: inception
      path: s3://cortex-examples/tensorflow/image-classifier/inception/
    - name: iris
      path: s3://cortex-examples/tensorflow/iris-classifier/nn/
    - name: resnet50
      path: s3://cortex-examples/tensorflow/resnet50/
  ...
```

#### `handler.py`

```python
class Handler:
    def __init__(self, tensorflow_client, config):
        self.client = tensorflow_client

    def handle_post(self, payload, query_params):
        model_name = query_params["model"]
        model_input = preprocess(payload["url"])
        results = self.client.predict(model_input, model_name)
        predicted_label = postprocess(results)
        return {"label": predicted_label}
```

## Multi-model caching

Multi-model caching allows Nucleus to serve more models than would fit into its memory by keeping a specified number of models in memory (and disk) at a time. When the in-memory model limit is reached, the least recently accessed model is evicted from the cache. This can be useful when you have many models, and some models are frequently accessed while a larger portion of them are rarely used, or when running on smaller instances to control costs.

The model cache is a two-layer cache, configured by the following parameters in the `models` configuration:

* `cache_size` sets the number of models to keep in memory
* `disk_cache_size` sets the number of models to keep on disk (must be greater than or equal to `cache_size`)

Both of these fields must be specified, in addition to either the `dir` or `paths` field (which specifies the model paths, see [models](#models) for documentation). Multi-model caching is only supported if `processes` is set to 1 (the default value).

### Out of memory errors

Nucleus runs a background process every 10 seconds that counts the number of models in memory and on disk, and evicts the least recently used models if the count exceeds `cache_size` / `disk_cache_size`. If many new models are requested between executions of the process, there may be more models in memory and/or on disk than the configured `cache_size` or `disk_cache_size` limits which could lead to out of memory errors.

## Server-side batching

Server-side batching is the process of aggregating multiple real-time requests into a single batch execution, which increases throughput at the expense of latency. Inference is triggered when either a maximum number of requests have been received, or when a certain amount of time has passed since receiving the first request, whichever comes first. Once a threshold is reached, the handling function is run on the received requests and responses are returned individually back to the clients. This process is transparent to the clients.

The Python and TensorFlow handlers allow for the use of the following two fields in the `server_side_batching` configuration:

* `max_batch_size`: The maximum number of requests to aggregate before running inference. This is an instrument for controlling throughput. The maximum size can be achieved if `batch_interval_seconds` is long enough to collect `max_batch_size` requests.

* `batch_interval_seconds`: The maximum amount of time to spend waiting for additional requests before running inference on the batch of requests. If fewer than `max_batch_size` requests are received after waiting the full `batch_interval_seconds`, then inference will run on the requests that have been received. This is an instrument for controlling latency.

*Note: Server-side batching is not supported for Nucleus servers that use the gRPC protocol.*

*Note: Server-side batching is only supported on the `handle_post` method.*

### Python Handler

When using server-side batching with the Python handler, the arguments that are passed into your handler's  function will be lists: `payload` will be a list of payloads, `query_params` will be a list of query parameter dictionaries, and `headers` will be a list of header dictionaries. The lists will all have the same length, where a particular index across all arguments corresponds to a single request (i.e. `payload[2]`, `query_params[2]`, and `headers[2]` correspond to a single request). Your handle function must return a list of responses in the same order that they were received (i.e. the 3rd element in returned list must be the response associated with `payload[2]`).

### TensorFlow Handler

In order to use server-side batching with the TensorFlow handler, the only requirement is that model's graph must be built such that batches can be accepted as input/output. No modifications to your handler implementation are required.

The following is an example of how the input `x` and the output `y` of the graph could be shaped to be compatible with server-side batching:

```python
batch_size = None
sample_shape = [340, 240, 3] # i.e. RGB image
output_shape = [1000] # i.e. image labels

with graph.as_default():
    # ...
    x = tf.placeholder(tf.float32, shape=[batch_size] + sample_shape, name="input")
    y = tf.placeholder(tf.float32, shape=[batch_size] + output_shape, name="output")
    # ...
```

#### Troubleshooting

Errors will be encountered if the model hasn't been built for batching.

The following error is an example of what happens when the input shape doesn't accommodate batching - e.g. when its shape is `[height, width, 3]` instead of `[batch_size, height, width, 3]`:

```text
Batching session Run() input tensors must have at least one dimension.
```

Here is another example of setting the output shape inappropriately for batching - e.g. when its shape is `[labels]` instead of `[batch_size, labels]`:

```text
Batched output tensor has 0 dimensions.
```

The solution to these errors is to incorporate into the model's graph another dimension (a placeholder for batch size) placed on the first position for both its input and output.

The following is an example of how the input `x` and the output `y` of the graph could be shaped to be compatible with server-side batching:

```python
batch_size = None
sample_shape = [340, 240, 3] # i.e. RGB image
output_shape = [1000] # i.e. image labels

with graph.as_default():
    # ...
    x = tf.placeholder(tf.float32, shape=[batch_size] + sample_shape, name="input")
    y = tf.placeholder(tf.float32, shape=[batch_size] + output_shape, name="output")
    # ...
```

### Optimization

When optimizing for both throughput and latency, you will likely want keep the `max_batch_size` to a relatively small value. Even though a higher `max_batch_size` with a low `batch_interval_seconds` (when there are many requests coming in) can offer a significantly higher throughput, the overall latency could be quite large. The reason is that for a request to get back a response, it has to wait until the entire batch is processed, which means that the added latency due to the `batch_interval_seconds` can pale in comparison. For instance, let's assume that a single request takes 50ms, and that when the batch size is set to 128, the processing time for a batch is 1280ms (i.e. 10ms per sample). So while the throughput is now 5 times higher, it takes 1280ms + `batch_interval_seconds` to get back a response (instead of 50ms). This is the trade-off with server-side batching.

When optimizing for maximum throughput, a good rule of thumb is to follow these steps:

1. Determine the maximum throughput of one Nucleus server when `server_side_batching` is not enabled (same as if `max_batch_size` were set to 1). This can be done with a load test (make sure to set `max_replicas` to 1 to disable autoscaling).
1. Determine the highest `batch_interval_seconds` with which you are still comfortable for your application. Keep in mind that the batch interval is not the only component of the overall latency - the inference on the batch and the pre/post processing also have to occur.
1. Multiply the maximum throughput from step 1 by the `batch_interval_seconds` from step 2. The result is a number which you can assign to `max_batch_size`.
1. Run the load test again. If the inference fails with that batch size (e.g. due to running out of GPU or RAM memory), then reduce `max_batch_size` to a level that works (reduce `batch_interval_seconds` by the same factor).
1. Use the load test to determine the peak throughput of the Nucleus server. Multiply the observed throughput by the `batch_interval_seconds` to calculate the average batch size. If the average batch size coincides with `max_batch_size`, then it might mean that the throughput could still be further increased by increasing `max_batch_size`. If it's lower, then it means that `batch_interval_seconds` is triggering the inference before `max_batch_size` requests have been aggregated. If modifying both `max_batch_size` and `batch_interval_seconds` doesn't improve the throughput, then the service may be bottlenecked by something else (e.g. CPU, network IO, `processes`, `threads_per_process`, etc).


# Handler implementation

## HTTP

### Handler

```python
# initialization code and variables can be declared here in global scope

class Handler:
    def __init__(self, config):
        """(Required) Called once before the Nucleus server becomes available. Performs
        setup such as downloading/initializing the model or downloading a
        vocabulary.

        Args:
            config (required): Dictionary passed from Nucleus configuration (if
                specified). This may contain information on where to download
                the model and/or metadata.
        """
        pass

    def handle_<HTTP-VERB>(self, payload, query_params, headers):
        """(Required) Called once per request. Preprocesses the request payload
        (if necessary), runs workload, and postprocesses the workload output
        (if necessary).

        Args:
            payload (optional): The request payload (see below for the possible
                payload types).
            query_params (optional): A dictionary of the query parameters used
                in the request.
            headers (optional): A dictionary of the headers sent in the request.

        Returns:
            Result or a batch of results.
        """
        pass
```

Your `Handler` class can implement methods for each of the following HTTP methods: POST, GET, PUT, PATCH, DELETE. Therefore, the respective methods in the `Handler` definition can be `handle_post`, `handle_get`, `handle_put`, `handle_patch`, and `handle_delete`.

For proper separation of concerns, it is recommended to use the constructor's `config` parameter for information such as from where to download the model and initialization files, or any configurable model parameters. You define `config` in your Nucleus configuration, and it is passed through to your Handler's constructor.

Your Nucleus server can accept requests with different types of payloads such as `JSON`-parseable, `bytes` or `starlette.datastructures.FormData` data. See [HTTP requests](#http-requests) to learn about how headers can be used to change the type of `payload` that is passed into your handler method.

Your handler method can return different types of objects such as `JSON`-parseable, `string`, and `bytes` objects. See [HTTP responses](#http-responses) to learn about how to configure your handler method to respond with different response codes and content-types.

### Callbacks

A callback is a function that starts running in the background after the results have been sent back to the client. They are meant to be short-lived.

Each handler method of your class can implement callbacks. To do this, when returning the result(s) from your handler method, also make sure to return a 2-element tuple in which the first element are your results that you want to return and the second element is a callable object that takes no arguments.

You can implement a callback like in the following example:

```python
def handle_post(self, payload):
    def _callback():
        print("message that gets printed after the response is sent back to the user")
    return "results", _callback
```

### HTTP requests

The type of the `payload` parameter in `handle_<HTTP-VERB>(self, payload)` can vary based on the content type of the request. The `payload` parameter is parsed according to the `Content-Type` header in the request. Here are the parsing rules (see below for examples):

1. For `Content-Type: application/json`, `payload` will be the parsed JSON body.
1. For `Content-Type: multipart/form-data` / `Content-Type: application/x-www-form-urlencoded`, `payload` will be `starlette.datastructures.FormData` (key-value pairs where the values are strings for text data, or `starlette.datastructures.UploadFile` for file uploads; see [Starlette's documentation](https://www.starlette.io/requests/#request-files)).
1. For `Content-Type: text/plain`, `payload` will be a string. `utf-8` encoding is assumed, unless specified otherwise (e.g. via `Content-Type: text/plain; charset=us-ascii`)
1. For all other `Content-Type` values, `payload` will be the raw `bytes` of the request body.

Here are some examples:

#### JSON data

##### Making the request

```bash
curl http://localhost:8080/ \
    -X POST -H "Content-Type: application/json" \
    -d '{"key": "value"}'
```

##### Reading the payload

When sending a JSON payload, the `payload` parameter will be a Python object:

```python
class Handler:
    def __init__(self, config):
        pass

    def handle_post(self, payload):
        print(payload["key"])  # prints "value"
```

#### Binary data

##### Making the request

```bash
curl http://localhost:8080/ \
    -X POST -H "Content-Type: application/octet-stream" \
    --data-binary @object.pkl
```

##### Reading the payload

Since the `Content-Type: application/octet-stream` header is used, the `payload` parameter will be a `bytes` object:

```python
import pickle

class Handler:
    def __init__(self, config):
        pass

    def handle_post(self, payload):
        obj = pickle.loads(payload)
        print(obj["key"])  # prints "value"
```

Here's an example if the binary data is an image:

```python
from PIL import Image
import io

class Handler:
    def __init__(self, config):
        pass

    def handle_post(self, payload, headers):
        img = Image.open(io.BytesIO(payload))  # read the payload bytes as an image
        print(img.size)
```

#### Form data (files)

##### Making the request

```bash
curl http://localhost:8080/ \
    -X POST \
    -F "text=@text.txt" \
    -F "object=@object.pkl" \
    -F "image=@image.png"
```

##### Reading the payload

When sending files via form data, the `payload` parameter will be `starlette.datastructures.FormData` (key-value pairs where the values are `starlette.datastructures.UploadFile`, see [Starlette's documentation](https://www.starlette.io/requests/#request-files)). Either `Content-Type: multipart/form-data` or `Content-Type: application/x-www-form-urlencoded` can be used (typically `Content-Type: multipart/form-data` is used for files, and is the default in the examples above).

```python
from PIL import Image
import pickle

class Handler:
    def __init__(self, config):
        pass

    def handle_post(self, payload):
        text = payload["text"].file.read()
        print(text.decode("utf-8"))  # prints the contents of text.txt

        obj = pickle.load(payload["object"].file)
        print(obj["key"])  # prints "value" assuming `object.pkl` is a pickled dictionary {"key": "value"}

        img = Image.open(payload["image"].file)
        print(img.size)  # prints the dimensions of image.png
```

#### Form data (text)

##### Making the request

```bash
curl http://localhost:8080/ \
    -X POST \
    -d "key=value"
```

##### Reading the payload

When sending text via form data, the `payload` parameter will be `starlette.datastructures.FormData` (key-value pairs where the values are strings, see [Starlette's documentation](https://www.starlette.io/requests/#request-files)). Either `Content-Type: multipart/form-data` or `Content-Type: application/x-www-form-urlencoded` can be used (typically `Content-Type: application/x-www-form-urlencoded` is used for text, and is the default in the examples above).

```python
class Handler:
    def __init__(self, config):
        pass

    def handle_post(self, payload):
        print(payload["key"])  # will print "value"
```

#### Text data

##### Making the request

```bash
curl http://localhost:8080/ \
    -X POST -H "Content-Type: text/plain" \
    -d "hello world"
```

##### Reading the payload

Since the `Content-Type: text/plain` header is used, the `payload` parameter will be a `string` object:

```python
class Handler:
    def __init__(self, config):
        pass

    def handle_post(self, payload):
        print(payload)  # prints "hello world"
```

### HTTP responses

The response of your `handle_<HTTP-VERB>()` method may be:

1. A JSON-serializable object (*lists*, *dictionaries*, *numbers*, etc.)

1. A `string` object (e.g. `"class 1"`)

1. A `bytes` object (e.g. `bytes(4)` or `pickle.dumps(obj)`)

1. An instance of [starlette.responses.Response](https://www.starlette.io/responses/#response)

## gRPC

To run your Nucleus server using the gRPC protocol, make sure the `protobuf_path` field in your Nucleus configuration is pointing to a protobuf file. When the server is started, Nucleus will compile the protobuf file for its use when running the server.

### Python Handler

#### Interface

```python
# initialization code and variables can be declared here in global scope

class Handler:
    def __init__(self, config, module_proto_pb2):
        """(Required) Called once before the Nucleus server becomes available. Performs
        setup such as downloading/initializing the model or downloading a
        vocabulary.

        Args:
            config (required): Dictionary passed from Nucleus configuration (if
                specified). This may contain information on where to download
                the model and/or metadata.
            module_proto_pb2 (required): Loaded Python module containing the
                class definitions of the messages defined in the protobuf
                file (`protobuf_path`).
        """
        self.module_proto_pb2 = module_proto_pb2

    def <RPC-METHOD-NAME>(self, payload, context):
        """(Required) Called once per request. Preprocesses the request payload
        (if necessary), runs workload, and postprocesses the workload output
        (if necessary).

        Args:
            payload (optional): The request payload (see below for the possible
                payload types).
            context (optional): gRPC context.

        Returns:
            Result (when streaming is not used).

        Yield:
            Result (when streaming is used).
        """
        pass
```

Your `Handler` class must implement the RPC methods found in the protobuf. Your protobuf must have a single service defined, which can have any name. If your service has 2 RPC methods called `Info` and `Predict` methods, then your `Handler` class must also implement these methods like in the above `Handler` template.

For proper separation of concerns, it is recommended to use the constructor's `config` parameter for information such as from where to download the model and initialization files, or any configurable model parameters. You define `config` in your Nucleus configuration, and it is passed through to your Handler class' constructor.

Your Nucleus server can only accept the type that has been specified in the protobuf definition of your service's method. See [gRPC requests](#grpc-requests) for how to construct gRPC requests.

Your handler method(s) can only return the type that has been specified in the protobuf definition of your service's method(s). See [gRPC responses](#grpc-responses) for how to handle gRPC responses.

### gRPC requests

Assuming the following service:

```protobuf
# handler.proto

syntax = "proto3";
package sample_service;

service Handler {
    rpc Predict (Sample) returns (Response);
}

message Sample {
    string a = 1;
}

message Response {
    string b = 1;
}
```

The handler implementation will also have a corresponding `Predict` method defined that represents the RPC method in the above protobuf service. The name(s) of the RPC method(s) is not enforced by Nucleus.

The type of the `payload` parameter passed into `Predict(self, payload)` will match that of the `Sample` message defined in the `protobuf_path` file. For this example, we'll assume that the above protobuf file was specified for the Nucleus server.

#### Simple request

The service method must look like this:

```protobuf
...
rpc Predict (Sample) returns (Response);
...
```

##### Making the request

```python
import grpc, handler_pb2, handler_pb2_grpc

stub = handler_pb2_grpc.HandlerStub(grpc.insecure_channel("http://localhost:8080/"))
stub.Predict(handler_pb2.Sample(a="text"))
```

##### Reading the payload

In the `Predict` method, you'll read the value like this:

```python
...
def Predict(self, payload):
    print(payload.a)
...
```

#### Streaming request

The service method must look like this:

```protobuf
...
rpc Predict (stream Sample) returns (Response);
...
```

##### Making the request

```python
import grpc, handler_pb2, handler_pb2_grpc

def generate_iterator(sample_list):
    for sample in sample_list:
        yield sample

stub = handler_pb2_grpc.HandlerStub(grpc.insecure_channel("http://localhost:8080/"))
stub.Predict(handler_pb2.Sample(generate_iterator(["a", "b", "c", "d"])))
```

##### Reading the payload

In the `Predict` method, you'll read the streamed values like this:

```python
...
def Predict(self, payload):
    for item in payload:
        print(item.a)
...
```

### gRPC responses

Assuming the following service:

```protobuf
# handler.proto

syntax = "proto3";
package sample_service;

service Handler {
    rpc Predict (Sample) returns (Response);
}

message Sample {
    string a = 1;
}

message Response {
    string b = 1;
}
```

The handler implementation will also have a corresponding `Predict` method defined that represents the RPC method in the above protobuf service. The name(s) of the RPC method(s) is not enforced by Nucleus.

The type of the value that you return in your `Predict()` method must match the `Response` message defined in the `protobuf_path` file. For this example, we'll assume that the above protobuf file was specified for the Nucleus server.

#### Simple response

The service method must look like this:

```protobuf
...
rpc Predict (Sample) returns (Response);
...
```

##### Making the request

```python
import grpc, handler_pb2, handler_pb2_grpc

stub = handler_pb2_grpc.HandlerStub(grpc.insecure_channel("http://localhost:8080/"))
r = stub.Predict(handler_pb2.Sample())
```

##### Returning the response

In the `Predict` method, you'll return the value like this:

```python
...
def Predict(self, payload):
    return self.proto_module_pb2.Response(b="text")
...
```

#### Streaming response

The service method must look like this:

```protobuf
...
rpc Predict (Sample) returns (stream Response);
...
```

##### Making the request

```python
import grpc, handler_pb2, handler_pb2_grpc

def generate_iterator(sample_list):
    for sample in sample_list:
        yield sample

stub = handler_pb2_grpc.HandlerStub(grpc.insecure_channel("http://localhost:8080/"))
for r in stub.Predict(handler_pb2.Sample())):
    print(r.b)
```

##### Returning the response

In the `Predict` method, you'll return the streamed values like this:

```python
...
def Predict(self, payload):
    for text in ["a", "b", "c", "d"]:
        yield self.proto_module_pb2.Response(b=text)
...
```

## Nucleus parallelism

Nucleus server parallelism can be configured with the following fields in the Nucleus configuration:

* `processes` (default: 1): Each Nucleus runs a web server with `processes` processes. For Nucleus servers with multiple CPUs, using 1-3 processes per unit of CPU generally leads to optimal throughput. For example, if `cpu` is 2, a value between 2 and 6 `processes` is reasonable. The optimal number will vary based on the workload's characteristics and the CPU compute request for the Nucleus server.

* `threads_per_process` (default: 1): Each process uses a thread pool of size `threads_per_process` to process requests. For applications that are not CPU intensive such as high I/O (e.g. downloading files) or GPU-based inference, increasing the number of threads per process can increase throughput. For CPU-bound applications such as running your model inference on a CPU, using 1 thread per process is recommended to avoid unnecessary context switching. Some applications are not thread-safe, and therefore must be run with 1 thread per process.

`processes` * `threads_per_process` represents the total number of requests that your Nucleus server can work on concurrently. For example, if `processes` is 2 and `threads_per_process` is 2, and the Nucleus server gets hit with 5 concurrent requests, 4 would immediately begin to be processed, and 1 would be waiting for a thread to become available. If the Nucleus server were to be hit with 3 concurrent requests, all three would begin processing immediately.

## Models

Live model reloading is a mechanism that periodically checks for updated models in the model path(s) provided in `models`. It is automatically enabled for all handler types, including the Python handler type (as long as model paths are specified via `multi_model_reloading` in the Nucleus configuration).

The following is a list of events that will trigger Nucleus to update its model(s):

* A new model is added to the model directory.
* A model is removed from the model directory.
* A model changes its directory structure.
* A file in the model directory is updated in-place.

### Python Handler

To use live model reloading with the Python handler, the model path(s) must be specified in the Nucleus configuration, via the `multi_model_reloading` field (this field is not required when not using live model reloading). When models are specified in this manner, your `Handler` class must implement the `load_model()` function, and models can be retrieved by using the `get_model()` method of the `model_client` that's passed into your handler's constructor.

#### Example

```python
class Handler:
    def __init__(self, config, model_client):
        self.client = model_client

    def load_model(self, model_path):
        # model_path is a path to your model's directory on disk
        return load_from_disk(model_path)

    def handle_post(self, payload):
      model = self.client.get_model()
      return model.predict(payload)
```

When multiple models are being served with Nucleus, `model_client.get_model()` can accept a model name:

```python
class Handler:
    # ...

    def handle_post(self, payload, query_params):
      model = self.client.get_model(query_params["model"])
      return model.predict(payload)
```

`model_client.get_model()` can also accept a model version if a version other than the highest is desired:

```python
class Handler:
    # ...

    def handle_post(self, payload, query_params):
      model = self.client.get_model(query_params["model"], query_params["version"])
      return model.predict(payload)
```

#### Interface

```python
# initialization code and variables can be declared here in global scope

class Handler:
    def __init__(self, config, model_client):
        """(Required) Called once before the Nucleus server becomes available. Performs
        setup such as downloading/initializing the model or downloading a
        vocabulary.

        Args:
            config (required): Dictionary passed from Nucleus configuration (if
                specified). This may contain information on where to download
                the model and/or metadata.
            model_client (required): Python client which is used to retrieve
                models for prediction. This should be saved for use in the handler method.
                Required when `multi_model_reloading` is specified in
                the Nucleus configuration.
        """
        self.client = model_client

    def load_model(self, model_path):
        """Called by Nucleus to load a model when necessary.

        This method is required when `multi_model_reloading`
        field is specified in the Nucleus configuration.

        Warning: this method must not make any modification to the model's
        contents on disk.

        Args:
            model_path: The path to the model on disk.

        Returns:
            The loaded model from disk. The returned object is what
            self.client.get_model() will return.
        """
        pass

    # define any handler methods for HTTP/gRPC workloads here
```

When explicit model paths are specified in the Python handler's Nucleus configuration, Nucleus provides a `model_client` to your Handler's constructor. `model_client` is an instance of [ModelClient](https://github.com/cortexlabs/nucleus/tree/master/src/cortex/cortex_internal/lib/client/python.py) that is used to load model(s) (it calls the `load_model()` method of your handler, which must be defined when using explicit model paths). It should be saved as an instance variable in your handler class, and your handler method should call `model_client.get_model()` to load your model for inference. Preprocessing of the JSON/gRPC payload and postprocessing of predictions can be implemented in your handler method as well.

When multiple models are defined using the Handler's `multi_model_reloading` field, the `model_client.get_model()` method expects an argument `model_name` which must hold the name of the model that you want to load (for example: `self.client.get_model("text-generator")`). There is also an optional second argument to specify the model version.

#### `load_model` method

The `load_model()` method that you implement in your `Handler` can return anything that you need to make a prediction. There is one caveat: whatever the return value is, it must be unloadable from memory via the `del` keyword. The following frameworks have been tested to work:

* PyTorch (CPU & GPU)
* ONNX (CPU & GPU)
* Sklearn/MLFlow (CPU)
* Numpy (CPU)
* Pandas (CPU)
* Caffe (not tested, but should work on CPU & GPU)

Python data structures containing these types are also supported (e.g. lists and dicts).

The `load_model()` method takes a single argument, which is a path (on disk) to the model to be loaded. Your `load_model()` method is called behind the scenes by Nucleus when you call the `model_client`'s `get_model()` method. Nucleus is responsible for downloading your model from S3 onto the local disk before calling `load_model()` with the local path. Whatever `load_model()` returns will be the exact return value of `model_client.get_model()`. Here is the schema for `model_client.get_model()`:

```python
def get_model(model_name, model_version):
    """
    Retrieve a model for inference.

    Args:
        model_name (optional): Name of the model to retrieve (when multiple models are run with a Nucleus server).
            When models.paths is specified, model_name should be the name of one of the models listed in the Nucleus config.
            When models.dir is specified, model_name should be the name of a top-level directory in the models dir.
        model_version (string, optional): Version of the model to retrieve. Can be omitted or set to "latest" to select the highest version.

    Returns:
        The value that's returned by your handler's load_model() method.
    """
```

#### Specifying models

Whenever a model path is specified in an Nucleus configuration file, it should be a path to an S3 prefix which contains your exported model. Directories may include a single model, or multiple folders each with a single model (note that a "single model" need not be a single file; there can be multiple files for a single model). When multiple folders are used, the folder names must be integer values, and will be interpreted as the model version. Model versions can be any integer, but are typically integer timestamps. It is always assumed that the highest version number is the latest version of your model.

##### API spec

###### Single model

The most common pattern is to serve a single model per Nucleus server. The path to the model is specified in the `path` field in the `multi_model_reloading` configuration. For example:

```yaml
# nucleus.yaml

type: python
multi_model_reloading:
  path: s3://my-bucket/models/text-generator/
```

###### Multiple models

It is possible to serve multiple models from a single Nucleus server. The paths to the models are specified in the Nucleus configuration, either via the `multi_model_reloading.paths` or `multi_model_reloading.dir` field in the configuration. For example:

```yaml
# nucleus.yaml

type: python
multi_model_reloading:
  paths:
    - name: iris-classifier
      path: s3://my-bucket/models/text-generator/
    # ...
```

or:

```yaml
# nucleus.yaml

type: python
multi_model_reloading:
  dir: s3://my-bucket/models/
```

It is also not necessary to specify the `multi_model_reloading` section at all, since you can download and load the model in your handler's `__init__()` function. That said, it is necessary to use the `multi_model_reloading` field to take advantage of live model reloading or multi-model caching.

When using the `multi_model_reloading.paths` field, each path must be a valid model directory (see above for valid model directory structures).

When using the `multi_model_reloading.dir` field, the directory provided may contain multiple subdirectories, each of which is a valid model directory. For example:

```text
  s3://my-bucket/models/
  ├── text-generator
  |   └── * (model files)
  └── sentiment-analyzer
      ├── 24753823/
      |   └── * (model files)
      └── 26234288/
          └── * (model files)
```

In this case, there are two models in the directory, one of which is named "text-generator", and the other is named "sentiment-analyzer".

###### Structure

Any model structure is accepted. Here is an example:

```text
  s3://my-bucket/models/text-generator/
  ├── model.pkl
  └── data.txt
```

or for a versioned model:

```text
  s3://my-bucket/models/text-generator/
  ├── 1523423423/  (version number, usually a timestamp)
  |   ├── model.pkl
  |   └── data.txt
  └── 2434389194/  (version number, usually a timestamp)
      ├── model.pkl
      └── data.txt
```

### TensorFlow Handler

In addition to the [standard Python Handler](#python-handler), Nucleus also supports another handler called the TensorFlow handler, which can be used to run TensorFlow models exported as `SavedModel` models. When using the TensorFlow handler, the model path(s) must be specified in the Nucleus configuration, via the `models` field.

#### Example

```python
class Handler:
    def __init__(self, tensorflow_client, config):
        self.client = tensorflow_client

    def handle_post(self, payload):
      return self.client.predict(payload)
```

When multiple models are being served with Nucleus, `tensorflow_client.predict()` can accept a model name:

```python
class Handler:
    # ...

    def handle_post(self, payload, query_params):
      return self.client.predict(payload, query_params["model"])
```

`tensorflow_client.predict()` can also accept a model version if a version other than the highest is desired:

```python
class Handler:
    # ...

    def handle_post(self, payload, query_params):
      return self.client.predict(payload, query_params["model"], query_params["version"])
```

#### Interface

```python
class Handler:
    def __init__(self, tensorflow_client, config):
        """(Required) Called once before the Nucleus server becomes available. Performs
        setup such as downloading/initializing a vocabulary.

        Args:
            tensorflow_client (required): TensorFlow client which is used to
                make predictions. This should be saved for use in the handler method.
            config (required): Dictionary passed from Nucleus configuration (if
                specified).
        """
        self.client = tensorflow_client
        # Additional initialization may be done here

    # define any handler methods for HTTP/gRPC workloads here
```

Nucleus provides a `tensorflow_client` to your Handler's constructor. `tensorflow_client` is an instance of [TensorFlowClient](https://github.com/cortexlabs/nucleus/tree/master/src/cortex/cortex_internal/lib/client/tensorflow.py) that manages a connection to a TensorFlow Serving container to make predictions using your model. It should be saved as an instance variable in your Handler class, and your handler method should call `tensorflow_client.predict()` to make an inference with your exported TensorFlow model. Preprocessing of the JSON payload and postprocessing of predictions can be implemented in your handler method as well.

When multiple models are defined using the Handler's `models` field, the `tensorflow_client.predict()` method expects a second argument `model_name` which must hold the name of the model that you want to use for inference (for example: `self.client.predict(payload, "text-generator")`). There is also an optional third argument to specify the model version.

If you need to share files between your handler implementation and the TensorFlow Serving container, you can create a new directory within `/mnt` (e.g. `/mnt/user`) and write files to it. **The entire `/mnt` directory is supposed to be shared between containers**, but do not write to any of the directories in `/mnt` that already exist (they are used internally by Nucleus).

#### `predict` method

Inference is performed by using the `predict` method of the `tensorflow_client` that's passed to the handler's constructor:

```python
def predict(model_input, model_name, model_version) -> dict:
    """
    Run prediction.

    Args:
        model_input: Input to the model.
        model_name (optional): Name of the model to retrieve (when multiple models are run with Nucleus).
            When models.paths is specified, model_name should be the name of one of the models listed in the Nucleus config.
            When models.dir is specified, model_name should be the name of a top-level directory in the models dir.
        model_version (string, optional): Version of the model to retrieve. Can be omitted or set to "latest" to select the highest version.

    Returns:
        dict: TensorFlow Serving response converted to a dictionary.
    """
```

#### Specifying models

Whenever a model path is specified in a Nucleus configuration file, it should be a path to an S3 prefix which contains your exported model. Directories may include a single model, or multiple folders each with a single model (note that a "single model" need not be a single file; there can be multiple files for a single model). When multiple folders are used, the folder names must be integer values, and will be interpreted as the model version. Model versions can be any integer, but are typically integer timestamps. It is always assumed that the highest version number is the latest version of your model.

##### API spec

###### Single model

The most common pattern is to serve a single model per Nucleus server. The path to the model is specified in the `path` field in the `models` configuration. For example:

```yaml
# nucleus.yaml

type: tensorflow
models:
  path: s3://my-bucket/models/text-generator/
```

###### Multiple models

It is possible to serve multiple models from a single Nucleus server. The paths to the models are specified in the Nucleus configuration, either via the `models.paths` or `models.dir` field in the Nucleus configuration. For example:

```yaml
# nucleus.yaml

type: tensorflow
models:
  paths:
    - name: iris-classifier
      path: s3://my-bucket/models/text-generator/
    # ...
```

or:

```yaml
# nucleus.yaml

type: tensorflow
models:
  dir: s3://my-bucket/models/
```

When using the `models.paths` field, each path must be a valid model directory (see above for valid model directory structures).

When using the `models.dir` field, the directory provided may contain multiple subdirectories, each of which is a valid model directory. For example:

```text
  s3://my-bucket/models/
  ├── text-generator
  |   └── * (model files)
  └── sentiment-analyzer
      ├── 24753823/
      |   └── * (model files)
      └── 26234288/
          └── * (model files)
```

In this case, there are two models in the directory, one of which is named "text-generator", and the other is named "sentiment-analyzer".

##### Structure

###### On CPU/GPU

The model path must be a SavedModel export:

```text
  s3://my-bucket/models/text-generator/
  ├── saved_model.pb
  └── variables/
      ├── variables.index
      ├── variables.data-00000-of-00003
      ├── variables.data-00001-of-00003
      └── variables.data-00002-of-...
```

or for a versioned model:

```text
  s3://my-bucket/models/text-generator/
  ├── 1523423423/  (version number, usually a timestamp)
  |   ├── saved_model.pb
  |   └── variables/
  |       ├── variables.index
  |       ├── variables.data-00000-of-00003
  |       ├── variables.data-00001-of-00003
  |       └── variables.data-00002-of-...
  └── 2434389194/  (version number, usually a timestamp)
      ├── saved_model.pb
      └── variables/
          ├── variables.index
          ├── variables.data-00000-of-00003
          ├── variables.data-00001-of-00003
          └── variables.data-00002-of-...
```

## Structured logging

You can use the built-in logger in your handler implemention to log in JSON. This will enrich your logs with metadata, and you can add custom metadata to the logs by adding key value pairs to the `extra` key when using the logger. For example:

```python
...
from cortex_internal.lib.log import logger as cortex_logger

class Handler:
    def handle_post(self, payload):
        cortex_logger.info("received payload", extra={"payload": payload})
```

The dictionary passed in via the `extra` will be flattened by one level. e.g.

```text
{"asctime": "2021-01-19 15:14:05,291", "levelname": "INFO", "message": "received payload", "process": 235, "payload": "this movie is awesome"}
```

To avoid overriding essential metadata, please refrain from specifying the following extra keys: `asctime`, `levelname`, `message`, `labels`, and `process`. When used with a Cortex cluster, log lines greater than 5 MB in size will be ignored.
