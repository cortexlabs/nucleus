#!/usr/bin/env bash

# Copyright 2021 Cortex Labs, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import yaml
import sys


def main(model_server_config_path: str):
    with open(model_server_config_path, "r") as f:
        server_config = yaml.safe_load(f)

    if "py_version" not in server_config:
        server_config["py_version"] = "3.6.9"

    if "use_local_cortex_libs" not in server_config:
        server_config["use_local_cortex_libs"] = False

    if "log_level" not in server_config:
        server_config["log_level"] = "info"
    else:
        server_config["log_level"] = server_config["log_level"].lower()

    if "env" not in server_config:
        server_config["env"] = {}

    if "config" not in server_config:
        server_config["config"] = {}

    if "serve_port" not in server_config:
        server_config["serve_port"] = 8080

    if "processes" not in server_config:
        server_config["processes"] = 1
    if "threads_per_process" not in server_config:
        server_config["threads_per_process"] = 1
    if "max_replica_concurrency" not in server_config:
        server_config["max_replica_concurrency"] = 1

    if "dependencies" not in server_config:
        server_config["dependencies"] = {
            "pip": "requirements.txt",
            "conda": "conda-packages.txt",
            "shell": "dependencies.sh",
        }
    elif "pip" not in server_config["dependencies"]:
        server_config["dependencies"]["pip"] = "requirements.txt"
    elif "conda" not in server_config["dependencies"]:
        server_config["dependencies"]["conda"] = "conda-packages.txt"
    elif "shell" not in server_config["dependencies"]:
        server_config["dependencies"]["shell"] = "dependencies.sh"

    if "python_path" not in server_config:
        server_config["python_path"] = "."

    if "protobuf_path" not in server_config:
        server_config["protobuf_path"] = None

    models_field_name: str = None
    if "multi_model_reloading" in server_config and server_config["multi_model_reloading"]:
        models_field_name = "multi_model_reloading"
        server_config["models"] = None
    elif "models" in server_config and server_config["models"]:
        models_field_name = "models"
        server_config["multi_model_reloading"] = None
    else:
        server_config["models"] = None
        server_config["multi_model_reloading"] = None
    if models_field_name:
        if "signature_key" not in server_config[models_field_name]:
            server_config[models_field_name]["signature_key"] = None
        if "paths" in server_config[models_field_name]:
            for idx, model in enumerate(server_config[models_field_name]["paths"]):
                if "signature_key" not in model:
                    server_config[models_field_name]["paths"][idx]["signature_key"] = None
        if "paths" not in server_config[models_field_name]:
            server_config[models_field_name]["paths"] = None
        if "dir" not in server_config[models_field_name]:
            server_config[models_field_name]["dir"] = None
        if "path" not in server_config[models_field_name]:
            server_config[models_field_name]["path"] = None
        if "cache_size" not in server_config[models_field_name]:
            server_config[models_field_name]["cache_size"] = None
        if "disk_cache_size" not in server_config[models_field_name]:
            server_config[models_field_name]["disk_cache_size"] = None

    print(yaml.safe_dump(server_config, indent=2, sort_keys=False))


if __name__ == "__main__":
    main(sys.argv[1])
